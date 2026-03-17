import sys
import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Timer

from muon import MuonWithAuxAdam
from ops.model import Model
from data.dataset import create_dataloaders, tokenizer, device


# ==================== 配置 ====================
@dataclass
class Config:
    """模型和训练配置，可直接作为 args 传给 Model"""

    # 模型
    n_embd: int = 384
    head_size: int = 64
    vocab_size: int = 6400
    D_DECAY_LORA: int = 64
    D_AAA_LORA: int = 64
    D_MV_LORA: int = 32
    D_GATE_LORA: int = 96
    max_iter: int = 12
    f_tol: float = 1e-6
    # 训练
    lr: float = 1e-3
    batch_size: int = 10
    max_length: int = 32
    epochs: int = 100
    val_split: float = 0.1
    # 早停
    early_stop_patience: int = 10
    # 路径
    data_path: str = "data/test.jsonl"
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"


cfg = Config()


# ==================== 训练模块 ====================
class TrainingModule(L.LightningModule):
    def __init__(self, args, lr=3e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["args"])  # 保存超参到 checkpoint
        self.model = Model(args)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, prefix):
        input_ids, target_ids = batch
        output, _ = self(input_ids)

        logits_flat = output.view(-1, output.size(-1))
        targets_flat = target_ids.view(-1)

        # 1) Loss
        loss = self.criterion(logits_flat, targets_flat)

        # 2) 困惑度 (Perplexity)
        ppl = loss.exp().clamp(max=1e4)  # 防止 inf

        # 3) 准确率 (token-level accuracy)
        mask = targets_flat != -100
        if mask.any():
            preds = logits_flat.argmax(dim=-1)
            acc = (preds[mask] == targets_flat[mask]).float().mean()
        else:
            acc = torch.tensor(0.0, device=loss.device)

        # 4) Top-5 准确率
        if mask.any():
            top5 = logits_flat.topk(5, dim=-1).indices  # [N, 5]
            top5_correct = (top5[mask] == targets_flat[mask].unsqueeze(-1)).any(dim=-1)
            top5_acc = top5_correct.float().mean()
        else:
            top5_acc = torch.tensor(0.0, device=loss.device)

        # 统一 log
        sync = prefix == "val"
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=(prefix == "train"), on_epoch=True, sync_dist=sync)
        self.log(f"{prefix}_ppl", ppl, prog_bar=True, on_epoch=True, sync_dist=sync)
        self.log(f"{prefix}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=sync)
        self.log(f"{prefix}_top5_acc", top5_acc, on_epoch=True, sync_dist=sync)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        if loss.isnan():
            raise ValueError("损失中出现 NaN，训练终止")

        # 5) 梯度范数 — 在 backward 之后 optimizer step 之前自动被 gradient_clip 截断，
        #    这里手动记录未截断的梯度范数用于监控
        total_norm = sum(p.grad.norm() ** 2 for p in self.parameters() if p.grad is not None) ** 0.5
        self.log("grad_norm", total_norm, on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_train_epoch_start(self):
        self._epoch_start = time.time()

    def on_train_epoch_end(self):
        # 6) Epoch 耗时
        elapsed = time.time() - self._epoch_start
        self.log("epoch_time_sec", elapsed, on_epoch=True, prog_bar=False)

        # 7) 当前学习率
        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"] if opt else self.lr
        self.log("lr", current_lr, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.device.type == "cpu":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            block_matrix = [p for p in self.model.block.parameters() if p.ndim >= 2]
            block_other = [p for p in self.model.block.parameters() if p.ndim < 2]
            non_block = [*self.model.emb.parameters(), *self.model.ln_out.parameters(), *self.model.head.parameters()]

            optimizer = MuonWithAuxAdam(
                [
                    dict(params=block_matrix, use_muon=True, lr=0.02, weight_decay=0.01),
                    dict(params=block_other + non_block, use_muon=False, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.01),
                ]
            )

        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ==================== 训练 ====================
def train(resume_from: str = None):
    model = TrainingModule(cfg, lr=cfg.lr)
    train_loader, val_loader = create_dataloaders(
        cfg.data_path,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        val_split=cfg.val_split,
    )

    # 8) 模型参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数：{total_params:,} 总量 / {trainable:,} 可训练")

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        logger=[
            CSVLogger(cfg.log_dir, name="csv"),
            TensorBoardLogger(cfg.log_dir, name="tensorboard"),  # tensorboard --logdir logs/tensorboard
        ],
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg.ckpt_dir,
                filename="model-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stop_patience,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

    # 保存最终权重
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), f"{cfg.ckpt_dir}/final_model.pt")

    val_loss = trainer.callback_metrics.get("val_loss")
    print(f"训练完成！最终验证损失：{val_loss:.4f}" if val_loss else "训练完成！（无验证指标）")


# ==================== 推理 ====================
@torch.no_grad()
def generate(model, prompt, max_length=32, temperature=1.0):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_length):
        logits, _ = model(input_ids)
        next_token = (logits[:, -1, :] / temperature).argmax(-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


# ==================== 入口 ====================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        prompt = sys.argv[2] if len(sys.argv) > 2 else "你好"
        model = Model(cfg).to(device)
        ckpt = f"{cfg.ckpt_dir}/final_model.pt"
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        except FileNotFoundError:
            sys.exit(f"未找到模型文件：{ckpt}，请先训练模型")
        print(f"输入：{prompt}\n输出：{generate(model, prompt)}")

    elif len(sys.argv) > 1 and sys.argv[1] == "resume":
        # 断点续训：python main.py resume [checkpoint_path]
        ckpt_path = sys.argv[2] if len(sys.argv) > 2 else f"{cfg.ckpt_dir}/last.ckpt"
        print(f"从 {ckpt_path} 恢复训练...")
        train(resume_from=ckpt_path)

    else:
        train()

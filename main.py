import sys
import torch
import torch.nn as nn
from dataclasses import dataclass

from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

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


cfg = Config()


# ==================== 训练模块 ====================
class TrainingModule(L.LightningModule):
    def __init__(self, args, lr=3e-2):
        super().__init__()
        self.model = Model(args)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, prefix):
        input_ids, target_ids = batch
        output, _ = self(input_ids)

        logits_flat = output.view(-1, output.size(-1))
        targets_flat = target_ids.view(-1)
        loss = self.criterion(logits_flat, targets_flat)

        with torch.no_grad():
            ppl = loss.exp()  # 困惑度
            preds = logits_flat.argmax(dim=-1)
            mask = targets_flat != -100  # 忽略 padding
            acc = (preds[mask] == targets_flat[mask]).float().mean()  # token 准确率

            # top-5 准确率
            top5 = logits_flat.topk(5, dim=-1).indices
            top5_acc = (top5[mask] == targets_flat[mask].unsqueeze(-1)).any(-1).float().mean()

        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{prefix}_ppl", ppl, prog_bar=True, on_epoch=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True, on_epoch=True)
        self.log(f"{prefix}_top5_acc", top5_acc, on_epoch=True)

        # 训练时额外记录学习率
        if prefix == "train":
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        if loss.isnan():
            raise ValueError("损失中出现 NaN")
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        if self.device.type == "cpu":
            # CPU 测试：直接用 AdamW，不依赖分布式
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
def train():
    model = TrainingModule(cfg, lr=cfg.lr)
    train_loader, val_loader = create_dataloaders(
        "data/test.jsonl",
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        val_split=cfg.val_split,
    )

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        logger=CSVLogger("logs/", name="training"),
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints/",
                filename="model-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,  # 连续 10 个 epoch 验证损失不下降则停止
                mode="min",
                verbose=True,
            ),
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    torch.save(model.model.state_dict(), "checkpoints/final_model.pt")
    metrics = trainer.callback_metrics
    print(
        f"训练完成！"
        f"验证损失：{metrics.get('val_loss', 0):.4f} | "
        f"验证困惑度：{metrics.get('val_ppl', 0):.2f} | "
        f"验证准确率：{metrics.get('val_acc', 0):.2%} | "
        f"验证Top5准确率：{metrics.get('val_top5_acc', 0):.2%}"
    )


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
    # uv run main.py generate "你好"
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        prompt = sys.argv[2] if len(sys.argv) > 2 else "你好"
        model = Model(cfg).to(device)
        ckpt = "checkpoints/final_model.pt"
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        except FileNotFoundError:
            sys.exit(f"未找到模型文件：{ckpt}，请先训练模型")
        print(f"输入：{prompt}\n输出：{generate(model, prompt)}")
    else:
        train()

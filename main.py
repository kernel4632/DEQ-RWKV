"""训练与推理入口。

这个文件只做两件事：
1. 组装训练流程。
2. 提供最简单的命令行推理入口。

调用方式：
- 训练：`uv run main.py`
- 推理：`uv run main.py generate "你好"`

这里刻意把“配置、训练步骤、推理步骤、权重加载”拆开写，
这样读代码时不用在一个超长函数里来回跳。
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from dataclasses import dataclass
from ops.model import Model
from data.dataset import create_dataloaders, tokenizer, device

torch.manual_seed(42)  
# ==================== 配置 ====================
@dataclass
class Config:
    """模型和训练配置，可直接作为 args 传给 Model"""

    # 模型
    n_embd = 384  # 384/512
    head_size = 64  # 性价比
    vocab_size = 6400  # 来自tokenizer
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 32
    D_GATE_LORA = 96
    # DEQ
    max_iter = 12  # 迭代次数
    f_tol = 1e-6  # 收敛阈值
    # 训练
    lr = 1e-3
    batch_size = 10
    max_length = 32
    epochs = 100
    val_split = 0.1  # 验证集占比，10%


config = Config()


# ==================== 训练模块 ====================
class TrainingModule(L.LightningModule):
    def __init__(self, args, lr=3e-2):
        super().__init__()
        self.save_hyperparameters(ignore=["args"])
        self.lr = lr
        self.model = Model(args)
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
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

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
            from muon import MuonWithAuxAdam

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
    # 为 Muon 优化器初始化单进程分布式环境
    if torch.cuda.is_available():
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            dist.init_process_group(backend="nccl", rank=0, world_size=1)

    model = TrainingModule(config, lr=config.lr)
    train_loader, val_loader = create_dataloaders(
        "data/test.jsonl",
        batch_size=config.batch_size,
        max_length=config.max_length,
        val_split=config.val_split,
    )
    ckpt_path = "checkpoints/last.ckpt"
    resume_path = ckpt_path if os.path.exists(ckpt_path) else None

    trainer = L.Trainer(
        max_epochs=config.epochs,
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
                every_n_train_steps=100,  # 每 100 个 batch 也保存一次 last.ckpt
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,  # 连续 10 个 epoch 验证损失不下降则停止
                mode="min",
                verbose=True,
            ),
        ],
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)
    torch.save(model.model.state_dict(), "checkpoints/final_model.pt")
    metrics = trainer.callback_metrics
    print(
        f"训练完成！"
        f"训练损失：{metrics.get('train_loss', 0):.4f} | "
        f"训练困惑度：{metrics.get('train_ppl', 0):.2f} | "
        f"训练准确率：{metrics.get('train_acc', 0):.2%} | "
        f"训练Top5准确率：{metrics.get('train_top5_acc', 0):.2%} | "
        "\n"
        f"学习率：{metrics.get('lr', 0):.6f} | "
        f"验证损失：{metrics.get('val_loss', 0):.4f} | "
        f"验证困惑度：{metrics.get('val_ppl', 0):.2f} | "
        f"验证准确率：{metrics.get('val_acc', 0):.2%} | "
        f"验证Top5准确率：{metrics.get('val_top5_acc', 0):.2%}"
    )


# ==================== 推理 ====================
@torch.no_grad()
def generate(model, prompt, max_length=64, temperature=1.0):
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

        # 优先用 final_model.pt，没有则从 last.ckpt 提取
        pt_path = "checkpoints/final_model.pt"
        ckpt_path = "checkpoints/last.ckpt"

        model = Model(config).to(device)

        if os.path.exists(pt_path):
            model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        elif os.path.exists(ckpt_path):
            # 从 Lightning checkpoint 中提取模型权重
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            model.load_state_dict(state_dict)
        else:
            sys.exit("未找到任何模型文件，请先训练模型")

        print(f"输入：{prompt}\n输出：{generate(model, prompt)}")
    else:
        train()

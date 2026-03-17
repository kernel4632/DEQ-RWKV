import torch, types
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from muon import MuonWithAuxAdam

from ops.model import Model
from data.dataset import create_dataloaders, device


# ==================== 配置区 ====================
class Config:
    """模型和训练配置"""

    # 模型配置
    n_embd = 384
    head_size = 64
    vocab_size = 6400
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 32
    D_GATE_LORA = 96
    max_iter = 12
    f_tol = 1e-6

    # 训练配置
    lr = 1e-3
    batch_size = 10
    max_length = 32
    epochs = 100
    val_split = 0.1  # 验证集比例 10%


# ==================== 训练模块 ====================
class TrainingModule(L.LightningModule):
    """DEQ-RWKV 训练模块，封装模型和训练逻辑"""

    def __init__(self, args, lr=3e-4):
        super().__init__()
        self.model = Model(args)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, batch):
        input_ids, target_ids = batch
        output, info = self.model(input_ids)
        loss = self.criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        if torch.isnan(loss).any():
            raise ValueError("损失中出现 NaN")

        # 训练指标统一用 train_loss，方便日志和回调读取
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("学习率", self.lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)

        # 这里记录为 val_loss，和 checkpoint 与 scheduler 的 monitor 对齐
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # 取出主干块的参数，Muon 只对“矩阵型权重”更有效
        hidden_weights = [p for p in self.model.block.parameters() if p.ndim >= 2]

        # 主干块里的一维参数（如增益、偏置），不适合 Muon
        hidden_gains_biases = [p for p in self.model.block.parameters() if p.ndim < 2]

        # 词表嵌入、输出层、归一化层属于“非主干参数”，继续用 AdamW
        nonhidden_params = [
            *self.model.emb.parameters(),
            *self.model.ln_out.parameters(),
            *self.model.head.parameters(),
        ]

        # 分组：主干用 Muon；其它参数用 AdamW
        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=0.02,
                weight_decay=0.01,
            ),
            dict(
                params=hidden_gains_biases + nonhidden_params,
                use_muon=False,
                lr=self.lr,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        # 监控项必须和 validation_step 的 log 名字一致
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


# ==================== 训练函数 ====================
def train():
    """启动训练流程"""
    args = types.SimpleNamespace(**{k: v for k, v in Config.__dict__.items() if not k.startswith("_")})

    # 初始化模型
    model = TrainingModule(args, lr=Config.lr)

    # 准备数据（自动拆分训练集和验证集）
    train_loader, val_loader = create_dataloaders(
        "data/test.jsonl",
        batch_size=Config.batch_size,
        max_length=Config.max_length,
        val_split=Config.val_split,
    )

    # 配置日志记录
    logger = CSVLogger("logs/", name="training")

    # 配置检查点
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # 配置 Trainer
    trainer = L.Trainer(
        max_epochs=Config.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[checkpoint],
    )

    # 开始训练
    trainer.fit(model, train_loader, val_loader)

    # 保存最终模型
    torch.save(model.model.state_dict(), "checkpoints/final_model.pt")
    print(f"训练完成！最终验证损失：{trainer.callback_metrics['val_loss'].item():.4f}")

    return model, trainer


# ==================== 推理函数 ====================
@torch.no_grad()
def generate(model, prompt, max_length=32, temperature=1.0):
    """文本生成推理"""
    model.eval()
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained("tokenizer")

    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 生成
    for _ in range(max_length):
        output, _ = model(input_ids)
        next_token_logits = output[:, -1, :] / temperature
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # 解码输出
    generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return generated_text


# ==================== 主程序 ====================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # 推理模式：python main.py generate "你的提示词"
        prompt = sys.argv[2] if len(sys.argv) > 2 else "你好"

        # 加载模型
        args = types.SimpleNamespace(**{k: v for k, v in Config.__dict__.items() if not k.startswith("_")})
        model = Model(args).to(device)

        checkpoint_path = "checkpoints/final_model.pt"
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print(f"已加载模型：{checkpoint_path}")
        except FileNotFoundError:
            print(f"未找到模型文件：{checkpoint_path}，请先训练模型")
            sys.exit(1)

        # 生成文本
        result = generate(model, prompt)
        print(f"\n输入：{prompt}")
        print(f"输出：{result}")
    else:
        # 训练模式
        train()

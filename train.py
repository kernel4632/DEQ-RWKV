import torch
import types
import json
import random
import torch

from model import DEQRWKV
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast



def create_model_args():
    args = types.SimpleNamespace()
    args.n_embd = 384
    args.head_size = 64
    args.vocab_size = 6400
    args.D_DECAY_LORA = 64
    args.D_AAA_LORA = 64
    args.D_MV_LORA = 32
    args.D_GATE_LORA = 96
    args.max_iter = 32
    args.f_tol = 1e-6
    return args


args = create_model_args()
torch.serialization.add_safe_globals([types.SimpleNamespace])
cuda_available = torch.cuda.is_available()


class MyRWKV(pl.LightningModule):
    def __init__(self, model_args, initial_lr, final_lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = DEQRWKV(self.hparams.model_args)
        if cuda_available:
            self.model = self.model.to(dtype=torch.bfloat16)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
    
    def _autocast_context(self):
        """创建一致的autocast上下文管理器"""
        return autocast(
            enabled=cuda_available, 
            device_type="cuda" if cuda_available else "cpu"
        )

    def forward(self, input_ids):
        with self._autocast_context():
            return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        with self._autocast_context():
            output, _ = self(input_ids)
            loss = self.loss_fn(output.view(-1, output.size(-1)), labels.view(-1))

        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        print("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        with self._autocast_context():
            output, _ = self(input_ids)
            loss = self.loss_fn(output.view(-1, output.size(-1)), labels.view(-1))

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_val_loss = torch.stack(self.validation_step_outputs).mean()
            self.log("avg_val_loss", avg_val_loss, logger=True)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # 创建优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.initial_lr)

        # 动态学习率调度器 - 利用LambdaLR实现线性衰减
        def lr_lambda(current_step):
            if not hasattr(self, "_total_steps"):
                # 延迟计算总步数直到训练开始
                train_loader = self.trainer.datamodule.train_dataloader()
                self._total_steps = len(train_loader) * self.trainer.max_epochs

            # 计算学习率衰减比例因子
            progress = min(current_step / self._total_steps, 1.0)
            lr_ratio = 1.0 - progress * (1.0 - self.hparams.final_lr / self.hparams.initial_lr)
            return lr_ratio

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }


class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        # 返回input_ids和对应的labels（shifted by 1）
        return torch.tensor(token[:-1]), torch.tensor(token[1:])  


class TokenDataModule(LightningDataModule):
    def __init__(self, file_path, batch_size=16, val_ratio=0.1, seed=42):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.all_tokens = None
        self.train_tokens = None
        self.val_tokens = None

    def prepare_data(self):
        # 只在一个进程中执行，用于数据下载或预处理
        self.all_tokens = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.all_tokens.append(data["token"])

    def setup(self, stage=None):
        # 在每个进程中执行，用于数据集分割
        random.seed(self.seed)
        random.shuffle(self.all_tokens)

        split_idx = int(len(self.all_tokens) * (1 - self.val_ratio))
        self.train_tokens = self.all_tokens[:split_idx]
        self.val_tokens = self.all_tokens[split_idx:]

    def _create_dataloader(self, tokens, shuffle):
        # 创建统一配置的DataLoader
        dataset = TokenDataset(tokens)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=4 if cuda_available else 0,  # CPU训练时减少worker数量
            shuffle=shuffle,
            pin_memory=cuda_available,
            persistent_workers=cuda_available,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_tokens, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_tokens, shuffle=False)


if __name__ == "__main__":
    # 设置随机种子以确保实验可重复性
    random.seed(42)
    torch.manual_seed(42)
    if cuda_available:
        torch.cuda.manual_seed_all(42)
        
    # 初始化模型和数据模块
    model = MyRWKV(model_args=args, initial_lr=6e-4, final_lr=2e-5)
    dm = TokenDataModule(
        file_path="minimind_dataset/test.jsonl", 
        batch_size=1, 
        val_ratio=0.1,
        seed=42
    )

    # 配置回调函数
    callbacks = [
        # 基于验证损失保存最佳模型
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath="./train_results/checkpoints",
            filename="best-model",
        ),
        # 按训练步数保存检查点
        pl.callbacks.ModelCheckpoint(
            monitor="train_loss", 
            mode="min",
            save_top_k=3,
            every_n_train_steps=10,
            dirpath="./train_results/checkpoints/step",
            filename="checkpoint-{step}",
        ),
        # 早停机制防止过拟合
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
        ),
    ]

    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        precision="bf16" if cuda_available else "32",
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        logger=True,
        enable_checkpointing=True,
        default_root_dir="./train_results",
        callbacks=callbacks,
    )

    # 开始训练
    trainer.fit(model=model, datamodule=dm)

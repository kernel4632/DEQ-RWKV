import sys
import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from muon import MuonWithAuxAdam
from ops.model import Model
from data.dataset import create_dataloaders, tokenizer, device


# ==================== 配置 ====================
@dataclass
class Config:
    """
    模型和训练的所有超参数集中在这里管理。
    使用 @dataclass 装饰器后，Config() 实例自带 .n_embd、.lr 等属性，
    可以直接当作 args 传给 Model(args)，不需要额外转换。
    """

    # ---------- 模型结构参数 ----------
    n_embd: int = 384  # 词嵌入维度，决定模型的"宽度"
    head_size: int = 64  # 注意力头的维度
    vocab_size: int = 6400  # 词表大小，需要和 tokenizer 的词表一致
    D_DECAY_LORA: int = 64  # RWKV 中 decay 分支的 LoRA 秩
    D_AAA_LORA: int = 64  # RWKV 中 AAA 分支的 LoRA 秩
    D_MV_LORA: int = 32  # RWKV 中 MV 分支的 LoRA 秩
    D_GATE_LORA: int = 96  # RWKV 中 Gate 分支的 LoRA 秩
    max_iter: int = 12  # DEQ（深度均衡模型）的最大不动点迭代次数
    f_tol: float = 1e-6  # DEQ 不动点迭代的收敛容差，残差小于此值即停止迭代

    # ---------- 训练参数 ----------
    lr: float = 1e-3  # 初始学习率
    batch_size: int = 10  # 每个批次包含多少条样本
    max_length: int = 32  # 每条样本的最大 token 长度（超出部分会被截断）
    epochs: int = 100  # 最多训练多少个 epoch
    val_split: float = 0.1  # 从数据中拆出 10% 作为验证集

    # ---------- 早停参数 ----------
    early_stop_patience: int = 10  # 验证 loss 连续多少个 epoch 没有改善就停止训练

    # ---------- 文件路径 ----------
    data_path: str = "data/test.jsonl"  # 训练数据文件（JSONL 格式，每行一个 {"text": "..."} ）
    ckpt_dir: str = "checkpoints"  # 模型检查点保存目录
    log_dir: str = "logs"  # 训练日志保存目录（CSV + TensorBoard）


# 创建全局配置实例，后续所有地方都用这个
cfg = Config()


# ==================== 训练模块 ====================
class TrainingModule(L.LightningModule):
    """
    Lightning 训练模块，把模型、损失函数、优化器、训练/验证逻辑封装在一起。
    Lightning 会自动调用 training_step、validation_step、configure_optimizers 等方法，
    只需要定义"每一步做什么"，不用手写训练循环。
    """

    def __init__(self, args, lr=3e-4):
        super().__init__()
        # save_hyperparameters() 会把 lr 等参数自动保存到 checkpoint 文件里，
        # 这样加载模型时可以知道当初是用什么超参训练的。
        # ignore=["args"] 是因为 args 是个复杂对象，不适合序列化。
        self.save_hyperparameters(ignore=["args"])
        self.model = Model(args)
        self.lr = lr
        # ignore_index=-100 表示：target 中值为 -100 的位置不参与 loss 计算。
        # 这与 dataset 中对 padding 位置填充 -100 配合使用，避免 padding 干扰训练。
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, prefix):
        """
        训练和验证共用的计算逻辑。prefix 为 "train" 或 "val"，用于区分日志名称。
        这样写避免了 training_step 和 validation_step 里重复相同的代码。
        """
        input_ids, target_ids = batch
        # 前向传播：输入 token 序列，得到每个位置对词表的预测概率分布
        # output 形状: [batch_size, seq_len, vocab_size]
        output, _ = self(input_ids)

        # 展平成二维，因为 CrossEntropyLoss 要求输入是 [样本数, 类别数] 的形状
        # logits_flat: [batch_size * seq_len, vocab_size]
        # targets_flat: [batch_size * seq_len]
        logits_flat = output.view(-1, output.size(-1))
        targets_flat = target_ids.view(-1)

        # ---- 损失 (Loss) ----
        # 交叉熵损失：衡量模型预测的概率分布与真实 token 之间的差距，越小越好
        loss = self.criterion(logits_flat, targets_flat)

        # ---- 困惑度 (Perplexity, PPL) ----
        # PPL = exp(loss)，是语言模型最常用的评估指标。
        # 直觉理解：PPL=100 表示模型在每个位置"犹豫"于 100 个候选词之间。
        # 理想情况下 PPL 越低越好（接近 1 表示模型非常确定）。
        # clamp(max=1e4) 防止训练初期 loss 很大时 exp 溢出为 inf。
        ppl = loss.exp().clamp(max=1e4)

        # ---- Token 级准确率 (Accuracy) ----
        # 逐个 token 判断：模型预测概率最高的那个 token 是否就是正确答案。
        # mask 用于排除 padding 位置（target=-100 的位置不参与计算）。
        mask = targets_flat != -100
        if mask.any():
            preds = logits_flat.argmax(dim=-1)  # 取概率最大的 token 作为预测结果
            acc = (preds[mask] == targets_flat[mask]).float().mean()
        else:
            # 如果整个 batch 全是 padding（极端情况），准确率记为 0
            acc = torch.tensor(0.0, device=loss.device)

        # ---- Top-5 准确率 ----
        # 比普通准确率更宽松：只要正确答案出现在模型预测的前 5 名里就算对。
        # 这个指标能反映模型是否"差一点就猜对了"，比单纯的 top-1 准确率更有参考价值。
        if mask.any():
            top5 = logits_flat.topk(5, dim=-1).indices  # 取概率最高的 5 个 token 的索引
            # unsqueeze(-1) 把 targets 从 [N] 变成 [N,1]，方便和 [N,5] 逐行比较
            top5_correct = (top5[mask] == targets_flat[mask].unsqueeze(-1)).any(dim=-1)
            top5_acc = top5_correct.float().mean()
        else:
            top5_acc = torch.tensor(0.0, device=loss.device)

        # ---- 统一记录日志 ----
        # prog_bar=True 表示在终端进度条上显示该指标
        # on_step=True 表示每个 batch 都记录（仅训练时），on_epoch=True 表示每个 epoch 汇总
        # sync_dist=True 在多 GPU 训练时同步各卡的指标（验证时需要，训练时不需要）
        sync = prefix == "val"
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=(prefix == "train"), on_epoch=True, sync_dist=sync)
        self.log(f"{prefix}_ppl", ppl, prog_bar=True, on_epoch=True, sync_dist=sync)
        self.log(f"{prefix}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=sync)
        self.log(f"{prefix}_top5_acc", top5_acc, on_epoch=True, sync_dist=sync)

        return loss

    def training_step(self, batch, batch_idx):
        """
        每个训练 batch 调用一次。Lightning 会自动处理 backward + optimizer.step。
        """
        loss = self._step(batch, "train")

        # 如果 loss 变成 NaN，说明训练已经崩溃（通常是学习率太大或数据有问题），
        # 直接抛出异常终止，避免浪费时间训练出一个废模型。
        if loss.isnan():
            raise ValueError("损失中出现 NaN，训练终止")

        # ---- 梯度范数 (Gradient Norm) ----
        # 梯度范数反映了当前这一步参数更新的"力度"。
        # 如果梯度范数突然变得很大，说明出现了梯度爆炸，模型可能即将崩溃。
        # 如果梯度范数接近 0，说明出现了梯度消失，模型学不到东西。
        # 注意：这里记录的是梯度裁剪之前的原始值，Trainer 的 gradient_clip_val 会在之后裁剪。
        total_norm = sum(p.grad.norm() ** 2 for p in self.parameters() if p.grad is not None) ** 0.5
        self.log("grad_norm", total_norm, on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        每个验证 batch 调用一次。Lightning 会自动关闭梯度计算（torch.no_grad），
        所以不需要手动写 with torch.no_grad()。
        """
        return self._step(batch, "val")

    def on_train_epoch_start(self):
        """每个 epoch 开始前记录时间戳，用于计算 epoch 耗时。"""
        self._epoch_start = time.time()

    def on_train_epoch_end(self):
        """每个 epoch 结束后记录耗时和当前学习率。"""
        # ---- Epoch 耗时 ----
        # 监控每个 epoch 花了多少秒，方便估算总训练时间和发现性能瓶颈
        elapsed = time.time() - self._epoch_start
        self.log("epoch_time_sec", elapsed, on_epoch=True, prog_bar=False)

        # ---- 当前学习率 ----
        # ReduceLROnPlateau 会在验证 loss 不下降时自动降低学习率，
        # 记录当前 lr 可以观察 scheduler 是否触发了衰减。
        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"] if opt else self.lr
        self.log("lr", current_lr, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器。Lightning 会自动调用这个方法。

        为什么要区分 CPU 和 GPU？
        - Muon 优化器依赖 PyTorch 分布式通信（dist.get_world_size），在没有 GPU 的单机环境下无法使用。
        - 所以 CPU 测试阶段用通用的 AdamW，迁移到 GPU 环境后自动切换为 Muon。
        """
        if self.device.type == "cpu":
            # CPU 环境：使用标准 AdamW 优化器，不依赖分布式
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            # GPU 环境：使用 Muon + AdamW 混合优化策略
            # Muon 对矩阵型权重（>=2维，如线性层的 weight）更有效，
            # 而一维参数（如 LayerNorm 的 gain/bias）和嵌入层不适合 Muon，仍用 AdamW。
            block_matrix = [p for p in self.model.block.parameters() if p.ndim >= 2]
            block_other = [p for p in self.model.block.parameters() if p.ndim < 2]
            non_block = [*self.model.emb.parameters(), *self.model.ln_out.parameters(), *self.model.head.parameters()]

            optimizer = MuonWithAuxAdam(
                [
                    dict(params=block_matrix, use_muon=True, lr=0.02, weight_decay=0.01),
                    dict(params=block_other + non_block, use_muon=False, lr=self.lr, betas=(0.9, 0.95), weight_decay=0.01),
                ]
            )

        # ReduceLROnPlateau：当 val_loss 连续 patience 个 epoch 没有改善时，
        # 将学习率乘以 factor（这里是减半）。这是一种自适应调整学习率的策略。
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        # 返回格式是 Lightning 要求的字典格式：
        # - "monitor" 告诉 scheduler 监控哪个指标来决定是否降低学习率
        # - 必须和 validation_step 中 self.log 的名字完全一致
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ==================== 训练 ====================
def train(resume_from: str = None):
    """
    启动训练流程。

    参数:
        resume_from: 如果提供 checkpoint 路径，则从该断点恢复训练（包括 epoch、优化器状态等）。
                     如果为 None，则从头开始训练。
    """
    model = TrainingModule(cfg, lr=cfg.lr)
    train_loader, val_loader = create_dataloaders(
        cfg.data_path,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        val_split=cfg.val_split,
    )

    # ---- 模型参数量统计 ----
    # 训练前打印参数量，帮助确认模型规模是否符合预期。
    # "可训练"参数是实际会被优化器更新的参数（如果有冻结层，两者会不同）。
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数：{total_params:,} 总量 / {trainable:,} 可训练")

    trainer = L.Trainer(
        max_epochs=cfg.epochs,  # 最多训练多少个 epoch（可能被早停提前终止）
        accelerator="auto",  # 自动检测：有 GPU 用 GPU，没有就用 CPU
        devices=1,  # 使用 1 个设备（单卡训练）
        log_every_n_steps=1,  # 每个 batch 都记录日志（数据量小时建议设为 1）
        gradient_clip_val=0.5,  # 梯度裁剪阈值：梯度范数超过 0.5 时会被缩放，防止梯度爆炸
        logger=[
            # CSV 日志：训练指标保存为 CSV 文件，方便用 pandas 分析
            CSVLogger(cfg.log_dir, name="csv"),
            # TensorBoard 日志：支持训练过程中实时可视化
            # 启动方式：在终端运行 tensorboard --logdir logs/tensorboard
            # 然后浏览器打开 http://localhost:6006 即可看到 loss、准确率等曲线
            TensorBoardLogger(cfg.log_dir, name="tensorboard"),
        ],
        callbacks=[
            # ---- 模型检查点 ----
            # 每个 epoch 结束后，如果 val_loss 是目前最好的，就保存一份 checkpoint。
            # save_top_k=3 表示最多保留最好的 3 个，旧的自动删除，节省磁盘空间。
            # save_last=True 额外保存一个 last.ckpt，用于断点续训。
            ModelCheckpoint(
                dirpath=cfg.ckpt_dir,
                filename="model-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",  # "min" 表示 val_loss 越小越好
                save_top_k=3,
                save_last=True,
            ),
            # ---- 早停 (Early Stopping) ----
            # 如果 val_loss 连续 early_stop_patience 个 epoch 都没有改善，
            # 就提前终止训练，避免过拟合和浪费计算资源。
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stop_patience,
                mode="min",
                verbose=True,  # 触发早停时打印提示信息
            ),
            # ---- 学习率监控 ----
            # 自动把每个 epoch 的学习率记录到日志中，方便在 TensorBoard 里查看 lr 变化曲线
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    # 开始训练。如果 resume_from 不为 None，Lightning 会自动从 checkpoint 恢复
    # epoch 计数、优化器状态、scheduler 状态等，实现无缝断点续训。
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

    # ---- 保存最终模型权重 ----
    # 这里只保存模型本身的 state_dict（不含优化器状态），文件更小，适合部署推理。
    # 和上面的 checkpoint 不同：checkpoint 包含完整训练状态（用于续训），
    # 而这个 final_model.pt 只包含模型权重（用于推理）。
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), f"{cfg.ckpt_dir}/final_model.pt")

    # 安全地获取验证 loss，防止某些异常情况下指标不存在导致崩溃
    val_loss = trainer.callback_metrics.get("val_loss")
    print(f"训练完成！最终验证损失：{val_loss:.4f}" if val_loss else "训练完成！（无验证指标）")


# ==================== 推理 ====================
@torch.no_grad()  # 推理时不需要计算梯度，关闭梯度可以节省内存和加速
def generate(model, prompt, max_length=32, temperature=1.0):
    """
    文本生成函数：给定一段提示词，让模型续写后续内容。

    参数:
        model: 训练好的模型
        prompt: 提示词（中文或英文文本）
        max_length: 最多生成多少个新 token
        temperature: 温度参数，控制生成的随机性。
                     - temperature=1.0：正常采样
                     - temperature<1.0：更保守，倾向于高概率词
                     - temperature>1.0：更随机，更有创造性
                     这里用的是 argmax（贪心解码），temperature 只影响 logits 的缩放。
    """
    model.eval()  # 切换到评估模式，关闭 Dropout 等训练专用的行为

    # 将提示词编码为 token ID 序列，并加上 batch 维度 [1, seq_len]
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    eos_id = tokenizer.eos_token_id  # 结束符的 token ID

    for _ in range(max_length):
        # 前向传播，得到每个位置的预测 logits
        logits, _ = model(input_ids)
        # 只取最后一个位置的 logits（自回归生成：基于已有内容预测下一个词）
        # 除以 temperature 调整概率分布的"尖锐程度"
        next_token = (logits[:, -1, :] / temperature).argmax(-1, keepdim=True)
        # 把新生成的 token 拼接到序列末尾，作为下一轮的输入
        input_ids = torch.cat([input_ids, next_token], dim=1)
        # 如果生成了结束符，提前停止，不再继续生成无意义内容
        if eos_id is not None and next_token.item() == eos_id:
            break

    # 将 token ID 序列解码回人类可读的文本
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 根据命令行参数决定运行模式：
    #   python main.py              → 从头开始训练
    #   python main.py generate     → 用默认提示词 "你好" 生成文本
    #   python main.py generate "提示词" → 用指定提示词生成文本
    #   python main.py resume       → 从 last.ckpt 断点续训
    #   python main.py resume path  → 从指定 checkpoint 断点续训

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # ---- 推理模式 ----
        prompt = sys.argv[2] if len(sys.argv) > 2 else "你好"
        model = Model(cfg).to(device)
        ckpt = f"{cfg.ckpt_dir}/final_model.pt"
        try:
            # weights_only=True 是安全选项，防止加载恶意 pickle 文件
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        except FileNotFoundError:
            sys.exit(f"未找到模型文件：{ckpt}，请先训练模型")
        print(f"输入：{prompt}\n输出：{generate(model, prompt)}")

    elif len(sys.argv) > 1 and sys.argv[1] == "resume":
        # ---- 断点续训模式 ----
        # 默认从 last.ckpt 恢复（ModelCheckpoint 的 save_last=True 会自动生成这个文件）
        ckpt_path = sys.argv[2] if len(sys.argv) > 2 else f"{cfg.ckpt_dir}/last.ckpt"
        print(f"从 {ckpt_path} 恢复训练...")
        train(resume_from=ckpt_path)

    else:
        # ---- 正常训练模式 ----
        train()

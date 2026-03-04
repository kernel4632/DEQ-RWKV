from .block import Block
import torch
import torch.nn as nn
from torchdeq import get_deq


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_embd = args.n_embd
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.block = Block(args)
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # DEQ 求解器参数
        self.max_iter = args.max_iter
        self.f_tol = args.f_tol

        # 初始化 DEQ 求解器
        self.deq = get_deq(
            ift=True,  # True 开启训练模式下的反向传播
            f_solver="broyden",  # 前向求解器类型
            f_max_iter=self.max_iter,  # 前向迭代最大次数
            f_tol=self.f_tol,  # 前向迭代收敛阈值
            b_solver="broyden",  # 反向求解器类型
            b_max_iter=self.max_iter,  # 反向迭代最大次数
            b_tol=self.f_tol,  # 反向迭代收敛阈值
        )

    def forward(self, x):
        x = self.preprocess(x)

        # 设置索引参数，给DE用，内部不会改变
        self.block.cmix.idx = x
        x = self.emb(x)

        # 重置值残差参数
        self.block.tmix.v_first = None

        # 执行 DEQ 求解
        out, info = self.deq(self.block, x)

        # 获取最终输出（固定点解）
        x = out[-1]

        # 应用输出层归一化和投影
        x = self.ln_out(x)
        x = self.head(x)

        return x, info

    def preprocess(self, inputData):
        """将任意格式的输入数据标准化为 [B, T] 形状的张量"""
        # --- 卫语句：处理空值和无效输入 ---
        if inputData is None:
            return torch.empty(0, 0)

        if isinstance(inputData, torch.Tensor) and inputData.numel() == 0:
            return inputData

        # --- 转换：统一转为张量 ---
        if not isinstance(inputData, torch.Tensor):
            inputData = torch.tensor(inputData)

        # --- 整形：补齐或校验维度 ---
        if inputData.dim() == 0:
            inputData = inputData.unsqueeze(0).unsqueeze(0)  # 标量 -> [1, 1]
        elif inputData.dim() == 1:
            inputData = inputData.unsqueeze(0)  # [T] -> [1, T]
        elif inputData.dim() == 2:
            pass  # [B, T] 已是目标格式
        else:
            raise ValueError(f"输入维度 {inputData.dim()} 超出支持范围，仅支持 0~2 维")

        return inputData

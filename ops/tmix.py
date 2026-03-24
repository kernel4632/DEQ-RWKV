import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wkv import wkv
from .rosa import ROSA


class Tmix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        self.n_embd = args.n_embd
        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size

        # 验证注意力维度能否被头大小整除
        assert args.dim_att % self.head_size == 0, "注意力维度必须能被头大小整除"

        H = self.n_head
        N = self.head_size
        C = args.n_embd

        D_DECAY_LORA = args.D_DECAY_LORA
        D_AAA_LORA = args.D_AAA_LORA
        D_MV_LORA = args.D_MV_LORA
        D_GATE_LORA = args.D_GATE_LORA

        for name in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0"]:
            setattr(self, name, nn.Parameter(torch.empty(1, 1, C)))

        lora_shapes = {
            "w": (C, D_DECAY_LORA),
            "a": (C, D_AAA_LORA),
            "v": (C, D_MV_LORA),
            "g": (C, D_GATE_LORA),
        }
        for prefix, (d_in, d_out) in lora_shapes.items():
            setattr(self, f"{prefix}1", nn.Parameter(torch.empty(d_in, d_out)))
            setattr(self, f"{prefix}2", nn.Parameter(torch.empty(d_out, d_in)))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        # 组归一化层，注意这里使用了非标准的 epsilon 值
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # 特殊的 epsilon 值

        # 时间移位操作
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # ROSA 模块，idx由上层模块直接指定传递
        self.idx = None
        self.rosa = ROSA()
        self.rosa_emb = nn.Embedding(args.vocab_size, args.dim_att)
        self.rosa_gate_w = nn.Parameter(torch.empty(1, 1, C))  # 门控的混合系数

        # 参数初始化
        self.init_weights()

    def init_weights(self):
        D_DECAY_LORA = self.args.D_DECAY_LORA
        D_AAA_LORA = self.args.D_AAA_LORA
        D_MV_LORA = self.args.D_MV_LORA
        D_GATE_LORA = self.args.D_GATE_LORA

        C = self.n_embd

        # 初始化线性层权重
        for layer in [self.receptance, self.key, self.value]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("linear"))
        # 备选：nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("linear"))
        # 备选：nn.init.zeros_(layer.weight)
        # 现在改为1/sqrt(2*n_layer)
        nn.init.xavier_uniform_(self.output.weight, gain=math.sqrt(2.0 / (C + C)))

        # 初始化时间移位系数参数
        input_params = [self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g]
        for param in input_params:
            nn.init.normal_(param, mean=0.0, std=0.01)

        # 初始化核参数
        nn.init.normal_(self.k_k, mean=1.0, std=0.02)
        nn.init.normal_(self.k_a, mean=0.0, std=0.02)

        nn.init.normal_(self.r_k, mean=0, std=0.01)

        lora_biases = [self.w0, self.a0, self.v0]
        for param in lora_biases:
            nn.init.normal_(param, mean=0.0, std=0.02)

        def init_lora_param(param, in_dim, out_dim, nonlinearity="linear"):
            gain = nn.init.calculate_gain(nonlinearity)
            std = gain * math.sqrt(2.0 / (in_dim + out_dim))
            nn.init.normal_(param, mean=0.0, std=std)

        lora_inputs = [
            (self.w1, C, D_DECAY_LORA, "tanh"),
            (self.a1, C, D_AAA_LORA, "sigmoid"),
            (self.v1, C, D_MV_LORA, "linear"),
            (self.g1, C, D_GATE_LORA, "sigmoid"),
        ]
        lora_outputs = [
            (self.w2, D_DECAY_LORA, C, "linear"),
            (self.a2, D_AAA_LORA, C, "linear"),
            (self.v2, D_MV_LORA, C, "linear"),
            (self.g2, D_GATE_LORA, C, "linear"),
        ]

        # 应用 LoRA 参数初始化
        for param, in_dim, out_dim, nonlinearity in lora_inputs + lora_outputs:
            init_lora_param(param, in_dim, out_dim, nonlinearity)

        # 初始化第一个值存储参数，用于值残差连接，设定为None，且可以由外部重置
        self.v_first = None

        # ROSA 参数初始化
        nn.init.normal_(self.rosa_gate_w, mean=0.0, std=0.01)

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        xx = self.time_shift(x) - x

        # 应用时间移位系数到不同的计算路径
        x_mix = torch.stack([self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g], dim=0)
        xr, xw, xk, xv, xa, xg = (x + xx * m for m in x_mix)

        # 计算 receptance, 衰减权重, 键和值
        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)

        # 处理值残差连接
        if self.v_first is None:
            self.v_first = v.detach().clone()  # 存储第一层的值用于残差计算

        # 应用值残差
        v = v + (self.v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # 计算上下文学习率和门控
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # 上下文学习率
        g = torch.sigmoid(xg @ self.g1) @ self.g2  # 输出门控

        # 处理键
        kk = k * self.k_k  # 用于归一化的键
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)  # 归一化
        k = k * (1 + (a - 1) * self.k_a)  # 应用上下文学习率到键

        # 执行 WKV 计算
        x = wkv(r, w, k, v, -kk, kk * a)

        # 稳定一下
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # 参数梯度动力学？反正是来引导wkv模块里的参数学习的，加这个更好
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)

        # 用一下 ROSA
        rosa_ids = torch.tensor(self.rosa(self.idx), device=x.device, dtype=torch.long).clamp(min=0)
        rosa_emb = self.rosa_emb(rosa_ids)  # 过 embedding 表拿向量 (B, T, C)
        gate = torch.sigmoid(x * self.rosa_gate_w).mean(dim=-1, keepdim=True)
        x = x * (1 - gate) + rosa_emb * gate

        # 应用输出门控和线性变换
        x = self.output(x * g)
        return x

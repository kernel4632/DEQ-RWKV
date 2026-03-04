import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wkv import wkv


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

        self.x_r = nn.Parameter(torch.randn(1, 1, C))
        self.x_w = nn.Parameter(torch.randn(1, 1, C))
        self.x_k = nn.Parameter(torch.randn(1, 1, C))
        self.x_v = nn.Parameter(torch.randn(1, 1, C))
        self.x_a = nn.Parameter(torch.randn(1, 1, C))
        self.x_g = nn.Parameter(torch.randn(1, 1, C))

        self.w0 = nn.Parameter(torch.randn(1, 1, C))
        self.w1 = nn.Parameter(torch.randn(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.randn(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.randn(1, 1, C))
        self.a1 = nn.Parameter(torch.randn(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.randn(D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.randn(1, 1, C))
        self.v1 = nn.Parameter(torch.randn(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.randn(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.randn(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.randn(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.randn(1, 1, C))
        self.k_a = nn.Parameter(torch.randn(1, 1, C))
        self.r_k = nn.Parameter(torch.randn(H, N))

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        # 组归一化层，注意这里使用了非标准的 epsilon 值
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # 特殊的 epsilon 值

        # 时间移位操作
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # 参数初始化
        self.init_weights()

    def init_weights(self):
        D_DECAY_LORA = self.args.D_DECAY_LORA
        D_AAA_LORA = self.args.D_AAA_LORA
        D_MV_LORA = self.args.D_MV_LORA
        D_GATE_LORA = self.args.D_GATE_LORA

        C = self.n_embd

        # 初始化线性层权重
        linear_layers = {
            "receptance": self.receptance,
            "key": self.key,
            "value": self.value,
            "output": self.output,
        }
        for name, layer in linear_layers.items():
            if name in ["receptance", "key", "value"]:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("linear"))
            elif name == "output":
                # 可以选择设为0：nn.init.zeros_(layer.weight)
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("linear"))

        # 初始化时间移位系数参数
        input_params = [self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g]
        for param in input_params:
            nn.init.normal_(param, mean=0.0, std=0.01)

        # 初始化核参数
        kernel_params = [self.k_k, self.k_a]
        for param in kernel_params:
            nn.init.normal_(param, mean=0.0, std=0.02)

        nn.init.orthogonal_(self.r_k)

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

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        xx = self.time_shift(x) - x

        # 应用时间移位系数到不同的计算路径
        xr = x + xx * self.x_r  # receptance 路径
        xw = x + xx * self.x_w  # 衰减路径
        xk = x + xx * self.x_k  # 键路径
        xv = x + xx * self.x_v  # 值路径
        xa = x + xx * self.x_a  # 上下文学习率路径
        xg = x + xx * self.x_g  # 门控路径

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

        # 应用输出门控和线性变换
        x = self.output(x * g)
        return x

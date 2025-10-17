import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchdeq import get_deq

# 设置设备和优化选项
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    from wkv7 import RUN_CUDA_RWKV7g as wkv
else:
    # CPU 版本的 WKV 实现
    def wkv(r, w, k, v, a, b):
        HEAD_SIZE = 64
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE

        # 调整张量形状并转换为 float 类型以保证计算精度
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))

        out = torch.zeros((B, T, H, N), device=r.device)
        state = torch.zeros((B, H, N, N), device=r.device)

        # 循环计算每个时间步的输出
        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)

            # WKV 核心计算公式
            state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)

        return out.view(B, T, C)

class Tmix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
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
        H = self.n_head

        # 初始化线性层权重
        linear_layers = {
            "receptance": self.receptance,
            "key": self.key,
            "value": self.value,
            "output": self.output,
        }
        for name, layer in linear_layers.items():
            if name in ["receptance", "key", "value"]:
                nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("linear")
                )
            elif name == "output":
                # 可以选择设为0：nn.init.zeros_(layer.weight)
                nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("linear")
                )

        # 初始化时间移位系数参数
        input_params = [self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g]
        for param in input_params:
            nn.init.normal_(param, mean=0.0, std=0.01)

        # 初始化核参数
        kernel_params = [self.k_k, self.k_a]
        for param in kernel_params:
            nn.init.normal_(param, mean=0.0, std=0.02)

        nn.init.normal_(self.r_k, mean=0.0, std=0.1)
        for h in range(H):
            self.r_k.data[h] += 0.1 * h

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

        # 初始化第一个值存储参数
        self.v_first = torch.zeros(1, 1, C, device=device)

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
        # 计算衰减权重 w，使用 softplus 进行软截断到 (-inf, -0.5]
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        # 处理值残差连接
        with torch.no_grad():
            if torch.all(self.v_first == 0):
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

        # 应用组归一化
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # 参数动力学？反正是来引导模型参数学习的
        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)

        # 应用输出门控和线性变换
        x = self.output(x * g)
        return x


class Cmix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_embd = args.n_embd
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.randn(1, 1, args.n_embd))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

        # 深度嵌入层：DE
        self.deepemb = nn.Embedding(args.vocab_size, args.n_embd * 4)

        # 参数初始化
        self.init_weights()

        # 索引参数，默认为0
        self.idx = torch.zeros(1, 1, device=device)

    def init_weights(self):
        nn.init.normal_(self.x_k, mean=0.0, std=1.0 / (self.n_embd**0.5))

        # 使用 He 初始化方法初始化前馈网络的权重
        nn.init.kaiming_uniform_(
            self.key.weight,
            mode="fan_in",
            nonlinearity="relu",
            a=0.5,  # LeakyReLU 的负斜率参数
        )

        # 线性输出层使用标准 He 初始化
        nn.init.kaiming_uniform_(
            self.value.weight, mode="fan_in", nonlinearity="linear"
        )

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        # 应用深度嵌入和线性变换
        return self.value(k * self.deepemb(self.idx))


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = Tmix(args)
        self.ffn = Cmix(args)

    def forward(self, x):
        # 重点注意：根据实验，发现此处不能使用残差连接，否则容易NaN
        x = x + self.att(self.ln1(x))
        x = self.ffn(
            self.ln2(x)
        )  
        return x


class DEQRWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.n_embd = args.n_embd

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.block = Block(args).to( device=device)
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
        ).to(device)

    def forward(self, x):
        x = self.preprocess(x)

        # 设置索引参数，给DE用
        self.block.ffn.idx = x
        x = self.emb(x)

        # 重置值残差参数
        self.block.att.v_first = torch.zeros_like(x, device=device)

        # 执行 DEQ 求解
        out, info = self.deq(self.block, x)

        # 获取最终输出（固定点解）
        x = out[-1]

        # 应用输出层归一化和投影
        x = self.ln_out(x)
        x = self.head(x)

        return x, info

    def preprocess(self, input):
        """预处理输入数据，确保其格式符合模型要求
        Args:
            input: 输入数据，可以是列表、numpy数组或torch.Tensor
        Returns:
            torch.Tensor: 处理后的张量，形状为 [B, T]
        Raises:
            ValueError: 当输入无法转换为张量或维度不正确时
        """
        # 确保输入是torch.Tensor类型
        if not isinstance(input, torch.Tensor):
            try:
                # 尝试将输入转换为张量
                input = torch.tensor(input)
            except Exception as e:
                raise ValueError(f"无法将输入转换为张量: {type(input)}, 错误: {str(e)}")

        # 确保输入在正确的设备上
        input = input.to(device)

        # 确保输入有正确的维度
        if input.dim() == 1:
            input = input.unsqueeze(0)  # 添加批次维度
        elif input.dim() > 2:
            raise ValueError(f"输入的维度不正确: {input.dim()}, 应为1或2维")

        return input

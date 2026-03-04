import torch
import torch.nn as nn


class Cmix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_embd = args.n_embd
        args.dim_ffn = args.n_embd * 4

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

        # 深度嵌入层：DE
        self.deepemb = nn.Embedding(args.vocab_size, args.dim_ffn)
        self.idx = None

        # 参数初始化
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.x_k, mean=0.0, std=1.0 / (self.n_embd**0.5))

        # 使用 He 初始化方法初始化前馈网络的权重
        nn.init.kaiming_uniform_(self.key.weight, mode="fan_in", nonlinearity="relu")

        # 线性输出层使用标准 He 初始化
        nn.init.kaiming_uniform_(self.value.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        x = torch.square(torch.relu(self.key(k)))

        # 应用深度嵌入和线性变换
        return self.value(x * self.deepemb(self.idx))

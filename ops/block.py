import torch.nn as nn
from .tmix import Tmix
from .cmix import Cmix


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.tmix = Tmix(args)
        self.cmix = Cmix(args)

    def forward(self, x):
        x = x + self.tmix(self.ln1(x))
        x = self.cmix(self.ln2(x))  # 注意：根据实验，发现此处不能像原版那样使用残差连接，否则容易NaN，造成原因不明
        return x

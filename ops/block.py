import torch
import torch.nn as nn
import torch.nn.functional as F
from .tmix import Tmix
from .cmix import Cmix
from .rosa import ROSA


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.tmix = Tmix(args)
        self.cmix = Cmix(args)

        self.rosa = ROSA()
        self.rosa_vocab_size = 64
        self.emb = nn.Embedding(self.rosa_vocab_size, args.n_embd)
        self.head = nn.Linear(args.n_embd, self.rosa_vocab_size, bias=False)
        self.gate = nn.Parameter(torch.empty(1, 1, args.n_embd))
        self.soft_gate = nn.Parameter(torch.empty(1, 1, args.n_embd))
        nn.init.normal_(self.gate, mean=0.0, std=0.01)
        nn.init.normal_(self.soft_gate, mean=0.0, std=0.01)
        self.idx = None

    def forward(self, x):
        if self.idx is not None:
            emb = self.emb(self.idx)
            gate = torch.sigmoid(x * self.gate).mean(dim=-1, keepdim=True)
            x = x * (1 - gate) + emb * gate
        # --------------------
        x = x + self.tmix(self.ln1(x))
        # 注意：根据实验，发现此处不能像原版那样使用残差连接，否则容易NaN，造成原因不明
        x = self.cmix(self.ln2(x))
        # --------------------
        logits = self.head(x)

        soft_prob = F.softmax(logits, dim=-1)
        soft_emb = soft_prob @ self.emb.weight
        soft_gate = torch.sigmoid(x * self.soft_gate).mean(dim=-1, keepdim=True)
        x = x * (1 - soft_gate) + soft_emb * soft_gate

        idx = torch.argmax(logits, dim=-1)
        idx = self.rosa(idx, return_type="tensor")
        self.idx = torch.as_tensor(idx, device=x.device, dtype=torch.long).clamp(min=0)

        return x

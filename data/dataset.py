# 导入必要的库
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tokenizer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokens(file_path, max_length=32):
    """从 JSONL 文件加载并编码所有文本"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [tokenizer.encode(json.loads(line)["text"])[:max_length] for line in f]


class TokenDataset(Dataset):
    """通用 token 数据集"""

    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        t = self.tokens[index]
        return torch.tensor(t[:-1]), torch.tensor(t[1:])


def _collate_fn(batch):
    """对变长序列做 padding 并移至设备"""
    inputs, targets = zip(*batch)
    pad_id = tokenizer.pad_token_id or 0
    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=-100)  # -100 让 CrossEntropyLoss 忽略 padding
    return inputs.to(device), targets.to(device)


def create_dataloader(file_path, batch_size=1, max_length=32, shuffle=True):
    """创建单个数据加载器"""
    dataset = TokenDataset(_load_tokens(file_path, max_length))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)


def create_dataloaders(file_path, batch_size=1, max_length=32, val_split=0.1, seed=42):
    """创建训练集和验证集数据加载器"""
    tokens = _load_tokens(file_path, max_length)

    random.seed(seed)
    random.shuffle(tokens)

    split_idx = int(len(tokens) * (1 - val_split))
    train_tokens, val_tokens = tokens[:split_idx], tokens[split_idx:]
    print(f"数据集拆分：训练集 {len(train_tokens)} 条，验证集 {len(val_tokens)} 条")

    train_loader = DataLoader(TokenDataset(train_tokens), batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(TokenDataset(val_tokens), batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    return train_loader, val_loader

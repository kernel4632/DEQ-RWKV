import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    """文本数据集，从 JSONL 文件加载数据"""

    def __init__(self, file_path, max_length=32):
        self.tokens = []
        self.max_length = max_length
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                token = tokenizer.encode(text)
                self.tokens.append(token[:max_length])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        input_ids = tokens[:-1]  # 输入为去掉最后一个 token
        target_ids = tokens[1:]  # 目标为去掉第一个 token
        return input_ids, target_ids


def _collate_fn(batch, device):
    """数据批处理函数"""
    inputs, targets = zip(*batch)
    inputs = torch.tensor(inputs, device=device)
    targets = torch.tensor(targets, device=device)
    return inputs, targets


def create_dataloader(file_path, batch_size=1, max_length=32, shuffle=True):
    """创建单个数据加载器"""
    dataset = TextDataset(file_path, max_length=max_length)

    def collate_fn(batch):
        return _collate_fn(batch, device)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)


def create_dataloaders(file_path, batch_size=1, max_length=32, val_split=0.1, seed=42):
    """创建训练集和验证集数据加载器

    Args:
        file_path: 数据文件路径
        batch_size: 批次大小
        max_length: 最大序列长度
        val_split: 验证集比例，默认 10%
        seed: 随机种子，保证可复现
    """
    # 加载数据
    tokens = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            token = tokenizer.encode(text)
            tokens.append(token[:max_length])

    # 打乱并拆分数据
    random.seed(seed)
    random.shuffle(tokens)

    split_idx = int(len(tokens) * (1 - val_split))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"数据集拆分：训练集 {len(train_tokens)} 条，验证集 {len(val_tokens)} 条")

    # 创建数据集
    class TokenDataset(Dataset):
        def __init__(self, tokens):
            self.tokens = tokens

        def __len__(self):
            return len(self.tokens)

        def __getitem__(self, index):
            tokens = self.tokens[index]
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            return input_ids, target_ids

    train_dataset = TokenDataset(train_tokens)
    val_dataset = TokenDataset(val_tokens)

    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: _collate_fn(batch, device), num_workers=0)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: _collate_fn(batch, device), num_workers=0)

    return train_loader, val_loader

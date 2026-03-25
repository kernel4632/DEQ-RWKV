import polars as pl
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import json
from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained("tokenizer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================== 修复后的 Polars 加载函数 ======================

def _load_tokens(
    file_path,
    max_length=32,
    min_score=0.75,  # 推荐范围：0.70~0.85，越高数据越精华
    sample_fraction=None,  # 采样比例，例如 0.2 = 采样20%
):
    """从 Parquet 或 JSONL 文件加载并编码文本"""
    file_path = Path(file_path)
    
    # 根据文件扩展名选择读取方式
    if file_path.suffix == ".parquet":
        lf = pl.scan_parquet(file_path)
        print("可用列:", lf.collect_schema().names())
        
        # 过滤高质量数据
        lf = lf.filter(pl.col("score") >= min_score).select(["text"])
        
        # 处理采样
        if sample_fraction is not None and sample_fraction < 1.0:
            df = lf.collect(engine="streaming")
            df = df.sample(fraction=sample_fraction, seed=42, shuffle=True)
            print(f"已随机采样 {sample_fraction * 100:.1f}% 的数据")
        else:
            df = lf.collect(engine="streaming")
            
        texts = df.get_column("text").to_list()
        
    elif file_path.suffix == ".jsonl":
        # 读取 JSONL 文件
        texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 检查是否有 score 字段且满足条件
                    if "score" in data and data["score"] < min_score:
                        continue
                    # 提取 text 字段
                    if "text" in data and isinstance(data["text"], str):
                        texts.append(data["text"])
                except json.JSONDecodeError:
                    continue
        
        # 处理采样
        if sample_fraction is not None and sample_fraction < 1.0:
            import random
            random.seed(42)
            sample_size = int(len(texts) * sample_fraction)
            texts = random.sample(texts, sample_size)
            print(f"已随机采样 {sample_fraction * 100:.1f}% 的数据")
    
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    # 编码成 tokens
    tokens = []
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        encoded = tokenizer.encode(text)[: max_length + 1]
        if len(encoded) > 1:  # 至少有两个 token（input + target）
            tokens.append(encoded)

    print(f"从 {file_path.name} 加载完成，共 {len(tokens)} 条有效序列 (score >= {min_score})")
    return tokens



# ====================== Dataset 和 collate_fn 保持不变 ======================
class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        t = self.tokens[index]
        return torch.tensor(t[:-1]), torch.tensor(t[1:])


def _collate_fn(batch):
    inputs, targets = zip(*batch)
    pad_id = tokenizer.pad_token_id or 0
    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return inputs.to(device), targets.to(device)


# ====================== dataloader 创建函数 ======================
def create_dataloaders(
    parquet_path,
    batch_size=1,
    max_length=32,
    val_split=0.1,
    seed=42,
    min_score=0.75,
    sample_fraction=None,
):
    """直接从 Parquet 创建 train/val dataloader"""

    tokens = _load_tokens(parquet_path, max_length=max_length, min_score=min_score, sample_fraction=sample_fraction)

    random.seed(seed)
    random.shuffle(tokens)

    split_idx = int(len(tokens) * (1 - val_split))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"数据集拆分：训练集 {len(train_tokens)} 条，验证集 {len(val_tokens)} 条")

    train_loader = DataLoader(TokenDataset(train_tokens), batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=_collate_fn)

    val_loader = DataLoader(TokenDataset(val_tokens), batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=_collate_fn)

    return train_loader, val_loader

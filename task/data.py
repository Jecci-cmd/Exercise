import random
import pandas as pd
from typing import List, Tuple

def generate_addition_problem(n_digits_1: int, n_digits_2: int) -> Tuple[str, str]:
    """生成一组加法题：输入字符串和输出结果字符串"""
    a = random.randint(10**(n_digits_1-1), 10**n_digits_1 - 1)
    b = random.randint(10**(n_digits_2-1), 10**n_digits_2 - 1)
    input_str = f"{a}+{b}"
    output_str = str(a + b)
    return input_str, output_str

def generate_dataset(n_samples: int, digit_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
    """生成一批样本，digit_pairs 形如 [(3,3), (3,4)]"""
    data = []
    for _ in range(n_samples):
        d1, d2 = random.choice(digit_pairs)
        inp, out = generate_addition_problem(d1, d2)
        data.append((inp, out))
    df = pd.DataFrame(data, columns=["input", "output"])
    return df

# 保存数据
def save_dataset(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"✅ 数据集已保存到 {path}，共 {len(df)} 条样本")

# 字符级 Tokenizer
class CharTokenizer:
    def __init__(self, data: pd.DataFrame):
        vocab = set("".join(data["input"].tolist() + data["output"].tolist()))
        self.vocab = sorted(list(vocab))
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.vocab = [self.pad_token, self.sos_token, self.eos_token] + self.vocab
        self.token2id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id2token = {i: ch for ch, i in self.token2id.items()}

    def encode(self, s: str, add_sos_eos=True) -> List[int]:
        ids = [self.token2id[c] for c in s]
        if add_sos_eos:
            ids = [self.token2id[self.sos_token]] + ids + [self.token2id[self.eos_token]]
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2token[i] for i in ids if i != self.token2id[self.pad_token]]
        if self.sos_token in tokens:
            tokens = tokens[tokens.index(self.sos_token)+1:]
        if self.eos_token in tokens:
            tokens = tokens[:tokens.index(self.eos_token)]
        return "".join(tokens)

    def vocab_size(self):
        return len(self.vocab)


# 生成数据
digit_combinations = [(3,3), (3,4), (4,3), (4,4), (3,5), (5,3)]
df = generate_dataset(n_samples=5000, digit_pairs=digit_combinations)
save_dataset(df, "addition_dataset.csv")

# 构造 tokenizer
tokenizer = CharTokenizer(df)
print("vocab:", tokenizer.vocab)
print("示例encode:", tokenizer.encode("123+456"))
print("示例decode:", tokenizer.decode(tokenizer.encode("123+456")))

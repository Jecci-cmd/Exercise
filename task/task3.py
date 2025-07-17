import os
import re
import math
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader,random_split

# 1.实现transformer
# (1) embedding + positional encoding
class PositionalEmbedding(nn.Module):
    def __init__(self,vocab_size,embed_dim,dropout=0.1,max_len=100):
        super(PositionalEmbedding,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros((1,max_len,embed_dim))
        position = torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,embed_dim,2).float() * (-math.log(10000.0) / embed_dim))
        pe[:,:,0::2] = torch.sin(position*div_term) 
        pe[:,:,1::2] = torch.cos(position*div_term) 
        self.register_buffer("pe",pe)
    def forward(self,x):
        x = self.embedding(x)
        x = x + self.pe[:,:x.size(1),:].to(x.device)
        return self.dropout(x) 

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.depth = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size = queries.shape[0]

        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)
        queries = queries.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.depth)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights 
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.W_o(output)


# (3) 前馈网络
class FeedForward(nn.Module):
    def __init__(self,embed_dim,num_hiddens,dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim,num_hiddens)
        self.linear2 = nn.Linear(num_hiddens,embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,num_hiddens,dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,    
            dropout=dropout
        )
        self.feed_forward = FeedForward(embed_dim,num_hiddens,dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        mid_output = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(mid_output))
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,num_hiddens,dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(embed_dim,num_hiddens,dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        mid_output = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(mid_output))
        
        cross_output = self.cross_attention(x, encoder_output, encoder_output, attn_mask=src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,layer,n_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])
    def forward(self,x,mask=None):
        for layer in self.layers:
            x = layer(x,mask)
        return x 

class TransformerDecoder(nn.Module):
    def __init__(self,layer,n_layers):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])
    def forward(self,x,encoder_output,src_mask=None,tgt_mask=None):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,embed_dim=128,num_heads=4,n_encoder_layers=3,n_decoder_layers=3,num_hiddens=256,dropout=0.1,max_len=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.src_embedding = PositionalEmbedding(src_vocab_size,embed_dim,dropout,max_len)
        self.tgt_embedding = PositionalEmbedding(tgt_vocab_size,embed_dim,dropout,max_len)
        decoder_layer = TransformerDecoderLayer(embed_dim,num_heads,num_hiddens,dropout)
        encoder_layer = TransformerEncoderLayer(embed_dim,num_heads,num_hiddens,dropout)
        self.decoder = TransformerDecoder(decoder_layer,n_decoder_layers)
        self.encoder = TransformerEncoder(encoder_layer,n_encoder_layers)
        self.output = nn.Linear(embed_dim,tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    def init_weights(self): 
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self,src,src_mask=None):
        src_emb = self.src_embedding(src)
        return self.encoder(src_emb,src_mask)
    def decode(self,tgt,encoder_output,src_mask=None,tgt_mask=None):
        tgt_emb = self.tgt_embedding(tgt)
        return self.decoder(tgt_emb,encoder_output,src_mask,tgt_mask)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output(decoder_output)


def create_src_mask(src, pad_idx=0):
    # 为源序列创建掩码（编码器输入和交叉注意力）
    # 形状: [batch_size, 1, 1, src_len]
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def create_tgt_mask(tgt, pad_idx=0):
    # 为目标序列创建组合掩码（解码器自注意力）,同时包含 padding 掩码和因果（causal）掩码
    # 形状: [batch_size, 1, tgt_len, tgt_len]
    # 形状: [batch_size, 1, 1, tgt_len]
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    # 形状: [tgt_len, tgt_len]
    device = tgt.device
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return tgt_mask

class DecoderOnlyLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,num_hiddens,dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(embed_dim,num_hiddens,dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attention(x, x, x, attn_mask=mask)
        # Add & Norm
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈网络
        ffn_output = self.feed_forward(x)
        # Add & Norm
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, n_layers, num_hiddens, dropout=0.1, max_len=256):
        super().__init__()
        self.max_len = max_len
        self.embedding = PositionalEmbedding(vocab_size, embed_dim, dropout, max_len)
        self.decoder_layers = nn.ModuleList([
            DecoderOnlyLayer(embed_dim, num_heads, num_hiddens, dropout) 
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.decoder_layers:
            x = layer(x, mask=mask)
        x = self.output_norm(x)
        logits = self.output(x)
        return logits
# 子任务1
class AdditionDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.build_vocab()
        self.tokenize_data()
  
    def build_vocab(self):
        all_text = []
        for _, row in self.data.iterrows():
            all_text.append(str(row['input']))
            all_text.append(str(row['output']))
        chars = set()
        for text in all_text:
            chars.update(text)
        self.vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for i, char in enumerate(sorted(list(chars))): 
            self.vocab[char] = i + 3   
        self.vocab_size = len(self.vocab)
        self.char_to_idx = self.vocab
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
    
    def tokenize_data(self):
        self.tokenized_data = []
        for _, row in self.data.iterrows():
            input_text = str(row['input'])
            output_text = str(row['output'])      
            input_tokens = [self.char_to_idx[char] for char in input_text]
            output_tokens = [self.char_to_idx['<SOS>']] + [self.char_to_idx[char] for char in output_text] + [self.char_to_idx['<EOS>']]
            self.tokenized_data.append((input_tokens, output_tokens))

    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def collate_fn(batch):
    inputs, outputs = zip(*batch)
    input_tensors = [torch.tensor(seq, dtype=torch.long) for seq in inputs]
    output_tensors = [torch.tensor(seq, dtype=torch.long) for seq in outputs]
    input_padded = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    output_padded = pad_sequence(output_tensors, batch_first=True, padding_value=0)
    return input_padded, output_padded

# def train_model(data_path='../data/addition_dataset.csv', embed_dim=128, num_heads=4, n_encoder_layers=3, n_decoder_layers=3,num_hiddens=256,dropout=0.1,batch_size=32,learning_rate=0.001,num_epochs=10,device='cuda' if torch.cuda.is_available() else 'cpu'):
#     dataset = AdditionDataset(data_path)
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#     model = Transformer(src_vocab_size=dataset.vocab_size,tgt_vocab_size=dataset.vocab_size,embed_dim=embed_dim,num_heads=num_heads,n_encoder_layers=n_encoder_layers,n_decoder_layers=n_decoder_layers,num_hiddens=num_hiddens,dropout=dropout,max_len=100).to(device) 
#     criterion = nn.CrossEntropyLoss(ignore_index=0)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     train_losses = []
#     train_accuracies = []
#     test_losses = []
#     test_accuracies = []    
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         correct_predictions = 0
#         total_predictions = 0
#         for batch_idx, (src, tgt) in enumerate(train_loader):
#             src, tgt = src.to(device), tgt.to(device)
#             tgt_input = tgt[:, :-1]
#             tgt_output = tgt[:, 1:]
            
#             src_mask = create_src_mask(src, pad_idx=0)
#             tgt_mask = create_tgt_mask(tgt_input, pad_idx=0)
            
#             optimizer.zero_grad()
#             output = model(src, tgt_input, src_mask, tgt_mask)
#             loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             pred = output.argmax(dim=-1)
#             mask = (tgt_output != 0)  
#             correct_predictions += ((pred == tgt_output) & mask).sum().item()
#             total_predictions += mask.sum().item()

#         avg_train_loss = total_loss / len(train_loader)
#         train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
#         model.eval()
#         total_test_loss = 0
#         test_correct_predictions = 0
#         test_total_predictions = 0
        
#         with torch.no_grad():
#             for src, tgt in test_loader:
#                 src, tgt = src.to(device), tgt.to(device)
#                 tgt_input = tgt[:, :-1]
#                 tgt_output = tgt[:, 1:]

#                 src_mask = create_src_mask(src, pad_idx=0)
#                 tgt_mask = create_tgt_mask(tgt_input, pad_idx=0)

#                 output = model(src, tgt_input, src_mask, tgt_mask)
#                 loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
#                 total_test_loss += loss.item()
#                 pred = output.argmax(dim=-1)
#                 mask = (tgt_output != 0)
#                 test_correct_predictions += ((pred == tgt_output) & mask).sum().item()
#                 test_total_predictions += mask.sum().item()
        
#         avg_test_loss = total_test_loss / len(test_loader)
#         test_accuracy = test_correct_predictions / test_total_predictions if test_total_predictions > 0 else 0
#         train_losses.append(avg_train_loss)
#         train_accuracies.append(train_accuracy)
#         test_losses.append(avg_test_loss)
#         test_accuracies.append(test_accuracy)
#         print(f'Epoch [{epoch+1}]')
#         print(f'  train loss: {avg_train_loss:.4f}, train acc: {train_accuracy:.4f}')
#         print(f'  test loss: {avg_test_loss:.4f}, test acc: {test_accuracy:.4f}')
    

#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='train loss')
#     plt.plot(test_losses, label='test loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(train_accuracies, label='train acc')
#     plt.plot(test_accuracies, label='test acc')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy Curve')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     test_examples(model, test_dataset, device, num_examples=100)
    
#     print("训练完成!")
#     return model, dataset

def test_examples(model, dataset, device, num_examples):
    model.eval()
    if isinstance(dataset, torch.utils.data.Subset):
        indices_to_sample = np.random.choice(len(dataset), num_examples, replace=False)
        original_indices = [dataset.indices[i] for i in indices_to_sample]
        base_dataset = dataset.dataset
    else:
        original_indices = np.random.choice(len(dataset), num_examples, replace=False)
        base_dataset = dataset

    results = []
    for idx in original_indices:
        input_tokens, target_tokens = base_dataset[idx]
        
        input_text = ''.join([base_dataset.idx_to_char[token] for token in input_tokens])
        target_text = ''.join([base_dataset.idx_to_char[token] for token in target_tokens[1:-1]])
        with torch.no_grad():
            src = torch.tensor([input_tokens], dtype=torch.long).to(device)
            predicted_text = generate_text(model, src, base_dataset, device, max_length=20)
        
        results.append({
            'input': input_text,
            'target': target_text,
            'predicted': predicted_text,
            'correct': target_text == predicted_text
        })
    
    print("\n测试例子:")
    print("=" * 60)
    for i, result in enumerate(results):
        status = "✓" if result['correct'] else "✗"
        print(f"{status} 输入: {result['input']}")
        print(f"  目标: {result['target']}")
        print(f"  预测: {result['predicted']}")
        print("-" * 40)
    
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    print(f"样本准确率: {accuracy:.2f}")

def generate_text(model, src, dataset, device, max_length=20):
    model.eval()
    src_mask = create_src_mask(src, pad_idx=0).to(device)
    encoder_output = model.encode(src, src_mask)
    tgt = torch.tensor([[dataset.char_to_idx['<SOS>']]], dtype=torch.long).to(device)
    
    for _ in range(max_length):
        tgt_mask = create_tgt_mask(tgt, pad_idx=0).to(device)
        
        with torch.no_grad():
            output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            logits = model.output(output)
        next_token = logits[0, -1, :].argmax().item()
        if next_token == dataset.char_to_idx['<EOS>']:
            break
        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
    generated_tokens = tgt[0, 1:].tolist()
    generated_text = ''.join([dataset.idx_to_char.get(token, '') for token in generated_tokens])
    
    return generated_text

# if __name__ == "__main__":
#     model, dataset = train_model(
#         data_path='../data/addition_dataset.csv',
#         embed_dim=128, 
#         num_heads=4,
#         n_encoder_layers=3,
#         n_decoder_layers=3,
#         num_hiddens=256,
#         dropout=0.1,
#         batch_size=32,
#         learning_rate=0.001,
#         num_epochs=10
#     )

# 2.子任务2
# class CharTokenizer:
#     def __init__(self, corpus):
#         self.chars = sorted(list(set(corpus)))
#         self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
#         self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

#     def encode(self, text):
#         return [self.char_to_idx[char] for char in text]

#     def decode(self, token_ids):
#         return "".join([self.idx_to_char[idx] for idx in token_ids])

#     @property
#     def vocab_size(self):
#         return len(self.chars)

# class LanguageModelDataset(Dataset):
#     def __init__(self, corpus, tokenizer, block_size):
#         self.tokenizer = tokenizer
#         self.block_size = block_size
#         self.tokens = tokenizer.encode(corpus)
        
#         self.examples = []
#         for i in range(0, len(self.tokens) - block_size):
#             self.examples.append(self.tokens[i : i + block_size + 1])

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         chunk = self.examples[idx]
#         x = torch.tensor(chunk[:-1], dtype=torch.long)
#         y = torch.tensor(chunk[1:], dtype=torch.long)
#         return x, y

# def generate_sequence(model, tokenizer, device, prompt, max_len=100):
#     model.eval()
#     tokens = tokenizer.encode(prompt)
#     x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

#     for _ in range(max_len):
#         current_input = x[:, -model.max_len:]
        
#         seq_len = current_input.size(1)
#         mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        
#         with torch.no_grad():
#             logits = model(current_input, mask=mask.unsqueeze(0).unsqueeze(0))
        
#         last_logits = logits[:, -1, :]
#         probs = F.softmax(last_logits, dim=-1)
#         next_token_id = torch.multinomial(probs, num_samples=1)
#         x = torch.cat([x, next_token_id], dim=1)
            
#     generated_ids = x[0].tolist()
#     generated_text = tokenizer.decode(generated_ids)
    
#     return generated_text

# def train_language_model(corpus_path,
#                          embed_dim=384,
#                          num_heads=6,
#                          n_layers=6,
#                          num_hiddens=384*4, # 通常是 embed_dim 的4倍
#                          dropout=0.2,
#                          batch_size=64,
#                          learning_rate=3e-4,
#                          num_epochs=10,
#                          block_size=256,
#                          device='cuda' if torch.cuda.is_available() else 'cpu'):
    
#     print("1. 加载和预处理数据...")
#     with open(corpus_path, 'r', encoding='utf-8') as f:
#         corpus = f.read()
    
#     corpus = corpus.lower()
    
#     tokenizer = CharTokenizer(corpus)
    
#     n = len(corpus)
#     train_corpus = corpus[:int(n*0.9)]
#     val_corpus = corpus[int(n*0.9):]

#     train_dataset = LanguageModelDataset(train_corpus, tokenizer, block_size)
#     val_dataset = LanguageModelDataset(val_corpus, tokenizer, block_size)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    
#     print(f"词汇表大小: {tokenizer.vocab_size}")
#     print(f"训练样本数: {len(train_dataset)}")
#     print(f"验证样本数: {len(val_dataset)}")
    
#     print("\n2. 初始化模型...")
#     model = DecoderOnlyTransformer(vocab_size=tokenizer.vocab_size,
#                                    embed_dim=embed_dim,
#                                    num_heads=num_heads,
#                                    n_layers=n_layers,
#                                    num_hiddens=num_hiddens,
#                                    dropout=dropout,
#                                    max_len=block_size).to(device)
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
#     print(f"\n3. 开始在 {device} 上训练...")
#     train_losses, val_losses = [], []
#     for epoch in range(num_epochs):
#         model.train()
#         total_train_loss = 0
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
            
#             seq_len = x.size(1)
#             mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

#             optimizer.zero_grad()
#             logits = model(x, mask=mask.unsqueeze(0).unsqueeze(0))
#             loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()
            
#         avg_train_loss = total_train_loss / len(train_loader)
#         train_losses.append(avg_train_loss)

#         # 验证
#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(device), y.to(device)
#                 seq_len = x.size(1)
#                 mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
#                 logits = model(x, mask=mask.unsqueeze(0).unsqueeze(0))
#                 loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
#                 total_val_loss += loss.item()
        
#         avg_val_loss = total_val_loss / len(val_loader)
#         val_losses.append(avg_val_loss)

#         print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

#         # 生成示例文本
#         prompt = "林轩道，"
#         generated_text = generate_sequence(model, tokenizer, device, prompt.lower(), max_len=200)
#         print(f"   生成示例: {generated_text}\n")
        
#     print("\n4. 训练完成!")
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Language Model Training & Validation Loss')
#     plt.legend()
#     plt.show()
    
#     return model, tokenizer

# if __name__ == "__main__":
#     corpus_path = '../data/tiny_shakespeare.txt'
#     lm_model, lm_tokenizer = train_language_model(corpus_path=corpus_path)

# 这是一个基于您原始代码的训练函数，只改变了返回值的形式
def run_single_training_original(data_path, embed_dim, num_heads, n_encoder_layers, n_decoder_layers,
                                 num_hiddens, dropout, batch_size, learning_rate, num_epochs,
                                 device='cuda' if torch.cuda.is_available() else 'cpu'):
    # --- 您原始代码的绝大部分都保持不变 ---
    dataset = AdditionDataset(data_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 这里的 Transformer 模型实例化将使用传入的参数
    model = Transformer(src_vocab_size=dataset.vocab_size, tgt_vocab_size=dataset.vocab_size,
                        embed_dim=embed_dim, num_heads=num_heads, n_encoder_layers=n_encoder_layers,
                        n_decoder_layers=n_decoder_layers, num_hiddens=num_hiddens, dropout=dropout,
                        max_len=100).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 我们只关心测试集的结果用于绘图对比
    test_losses_history = []
    test_accuracies_history = []

    for epoch in range(num_epochs):
        model.train()
        # --- 训练循环 (完全是您原始的代码) ---
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 使用您原始的掩码逻辑
            src_mask = create_src_mask(src, pad_idx=0)
            tgt_mask = create_tgt_mask(tgt_input, pad_idx=0)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask) # 调用您原始的 forward 方法
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
        
        # --- 评估循环 (完全是您原始的代码) ---
        model.eval()
        total_test_loss = 0
        test_correct_predictions = 0
        test_total_predictions = 0
        with torch.no_grad():
            for src, tgt in test_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # 使用您原始的掩码逻辑
                src_mask = create_src_mask(src, pad_idx=0)
                tgt_mask = create_tgt_mask(tgt_input, pad_idx=0)

                output = model(src, tgt_input, src_mask, tgt_mask) # 调用您原始的 forward 方法
                loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
                total_test_loss += loss.item()
                pred = output.argmax(dim=-1)
                mask = (tgt_output != 0)
                test_correct_predictions += ((pred == tgt_output) & mask).sum().item()
                test_total_predictions += mask.sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = test_correct_predictions / test_total_predictions if test_total_predictions > 0 else 0
        
        test_losses_history.append(avg_test_loss)
        test_accuracies_history.append(test_accuracy)
        
        print(f"  (LR={learning_rate}, Heads={num_heads}, Depth={n_encoder_layers}) Epoch {epoch+1} -> Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    # *** 关键改动：不再绘图，而是返回历史记录 ***
    return test_losses_history, test_accuracies_history

# (1)测试不同学习率
def test_learning_rates_simple(data_path, num_epochs=10):
    learning_rates_to_test = [0.01, 0.001, 0.0001]
    results = {}

    for lr in learning_rates_to_test:
        print(f"\n{'='*20} Testing Learning Rate: {lr} {'='*20}")
        # 固定其他参数，只改变学习率
        test_losses, test_accs = run_single_training_original(
            data_path=data_path, embed_dim=128, num_heads=4, n_encoder_layers=3, 
            n_decoder_layers=3, num_hiddens=256, dropout=0.1, batch_size=32, 
            learning_rate=lr, num_epochs=num_epochs
        )
        results[lr] = {'test_loss': test_losses, 'test_acc': test_accs}
    
    # 统一绘图
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for lr, data in results.items():
        plt.plot(data['test_loss'], label=f'LR = {lr}')
    plt.title('Test Loss vs. Epochs for Different Learning Rates')
    plt.xlabel('Epoch'); plt.ylabel('Test Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    for lr, data in results.items():
        plt.plot(data['test_acc'], label=f'LR = {lr}')
    plt.title('Test Accuracy vs. Epochs for Different Learning Rates')
    plt.xlabel('Epoch'); plt.ylabel('Test Accuracy'); plt.legend(); plt.grid(True)
    
    plt.tight_layout(); plt.show()

# (2) 测试不同注意力头数
def test_attention_heads_simple(data_path, num_epochs=10):
    num_heads_to_test = [2, 4, 8]
    results = {}

    for n_heads in num_heads_to_test:
        print(f"\n{'='*20} Testing Attention Heads: {n_heads} {'='*20}")
        # 固定其他参数，只改变注意力头数
        test_losses, test_accs = run_single_training_original(
            data_path=data_path, embed_dim=128, num_heads=n_heads, n_encoder_layers=3, 
            n_decoder_layers=3, num_hiddens=256, dropout=0.1, batch_size=32, 
            learning_rate=0.001, num_epochs=num_epochs
        )
        results[n_heads] = {'test_loss': test_losses, 'test_acc': test_accs}

    # 统一绘图
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for n_heads, data in results.items():
        plt.plot(data['test_loss'], label=f'Heads = {n_heads}')
    plt.title('Test Loss vs. Epochs for Different Attention Heads')
    plt.xlabel('Epoch'); plt.ylabel('Test Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    for n_heads, data in results.items():
        plt.plot(data['test_acc'], label=f'Heads = {n_heads}')
    plt.title('Test Accuracy vs. Epochs for Different Attention Heads')
    plt.xlabel('Epoch'); plt.ylabel('Test Accuracy'); plt.legend(); plt.grid(True)
    
    plt.tight_layout(); plt.show()

# (3) 不同模型深度
def test_model_depth_simple(data_path, num_epochs=10):
    depths_to_test = [2, 3, 4]
    results = {}

    for depth in depths_to_test:
        print(f"\n{'='*20} Testing Model Depth: {depth} {'='*20}")
        # 固定其他参数，只改变模型深度
        test_losses, test_accs = run_single_training_original(
            data_path=data_path, embed_dim=128, num_heads=4, n_encoder_layers=depth, 
            n_decoder_layers=depth, num_hiddens=256, dropout=0.1, batch_size=32, 
            learning_rate=0.001, num_epochs=num_epochs
        )
        results[depth] = {'test_loss': test_losses, 'test_acc': test_accs}
        
    # 统一绘图
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for depth, data in results.items():
        plt.plot(data['test_loss'], label=f'Depth = {depth}')
    plt.title('Test Loss vs. Epochs for Different Model Depths')
    plt.xlabel('Epoch'); plt.ylabel('Test Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    for depth, data in results.items():
        plt.plot(data['test_acc'], label=f'Depth = {depth}')
    plt.title('Test Accuracy vs. Epochs for Different Model Depths')
    plt.xlabel('Epoch'); plt.ylabel('Test Accuracy'); plt.legend(); plt.grid(True)
    
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    DATA_PATH = '../data/addition_dataset.csv'

    epoch = 5 

    print("--- 开始测试不同学习率 ---")
    test_learning_rates_simple(data_path=DATA_PATH, num_epochs=epoch)
    
    print("\n--- 开始测试不同注意力头数 ---")
    test_attention_heads_simple(data_path=DATA_PATH, num_epochs=epoch)
    
    print("\n--- 开始测试不同模型深度 ---")
    test_model_depth_simple(data_path=DATA_PATH, num_epochs=epoch)
import os
import re
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
glove_path = os.path.join('..','data','glove.6B.50d.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1.加载数据集
data_file = os.path.join('..','data','new_train.tsv')
data = pd.read_csv(data_file,sep='\t')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
# 2.划分训练集和验证集
X_train_pre,X_valid_pre,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)
y_train = torch.tensor(y_train).to(device)
y_valid = torch.tensor(y_valid).to(device)
test_file = os.path.join('..','data','new_test.tsv')
test = pd.read_csv(test_file,sep='\t')
X_test_pre = test.iloc[:,:-1].values
y_test = torch.tensor(test.iloc[:,-1].values).to(device)
# 3.数据集预处理
def preprocess_text(text):
    text = " ".join(text.flatten())
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    words = text.split()
    return words
X_train = [preprocess_text(text) for text in X_train_pre]
X_valid = [preprocess_text(text) for text in X_valid_pre]
X_test = [preprocess_text(text) for text in X_test_pre]
# 4.构建词典
def build_vocab_tokenizer(texts,min_freq=1):
    counter = Counter()
    for words in texts:
        counter.update(words)
    vocab = {'<PAD>':0,'<UNK>':1}
    for word,freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab
vocab = build_vocab_tokenizer(X_train)
# 5.把文本转为索引序列
def encode(texts,vocab):
    return [[vocab.get(word,vocab['<UNK>']) for word in sentence] for sentence in texts] # 嵌套列表推导式
X_train = encode(X_train,vocab)
X_valid = encode(X_valid,vocab)
X_test = encode(X_test,vocab)
# 6.补齐句子
def pad_truncate(texts,max_len=30,padding_value=0):
    padded = []
    for words in texts:
        words = torch.tensor(words)
        if len(words) > max_len:
            padded_seq = words[:max_len]
        else:
            padded_seq = torch.cat([words,torch.full((max_len-len(words),),padding_value)])
        padded.append(padded_seq)
    return torch.stack(padded)
X_train = pad_truncate(X_train).to(device)
X_valid = pad_truncate(X_valid).to(device)
X_test = pad_truncate(X_test).to(device)
# 7.定义模型
class TextCNN(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_classes,num_channels=100,kernel_size=[3,4,5],dropout=0.7):
        super(TextCNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim,out_channels=num_channels,kernel_size=k) for k in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels*len(kernel_size),num_classes)
    def forward(self,x): # x是batch_size,max_len
        x = self.embedding(x)
        x = x.permute(0,2,1) # batch_size,embed_dim,max_len,embed_dim相当于num_channels
        conv_outputs = []
        for conv in self.convs:
            c = conv(x) # batch_size,embed_dim,max_len - kernel_size + 1
            c = torch.relu(c)
            c = torch.max_pool1d(c,kernel_size=c.shape[2]) # batch_size,embed_dim,1
            conv_outputs.append(c.squeeze(2)) # batch_size,embed_dim
        x = torch.cat(conv_outputs,dim=1)  # batch_size,embed_dim*len(kernel_size)
        x = self.dropout(x)
        out = self.fc(x)
        return out
# 8.训练和测试函数
def train_evaluate(model,optimizer,epochs,X_train,y_train,X_valid,y_valid,loss_fc=torch.nn.CrossEntropyLoss(),batch_size=64,device="cuda"):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = (len(X_train) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = min(len(X_train),start + batch_size)
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fc(y_pred,y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        total_valid_loss = 0
        correct = 0
        total = 0
        valid_batches = (len(X_valid) + batch_size - 1) // batch_size
        with torch.no_grad():
            for j in range(valid_batches):
                start = j * batch_size
                end = min(len(X_valid), start + batch_size)
                X_val = X_valid[start:end].to(device)
                y_val = y_valid[start:end].to(device)

                y_val_pred = model(X_val)
                val_loss = loss_fc(y_val_pred, y_val)
                total_valid_loss += val_loss.item()

                preds = y_val_pred.argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
        print(f'epoch{epoch + 1} train Loss = {(total_loss / num_batches):.4f},valid loss = {(total_valid_loss / valid_batches):.4f},valid acc = {100*correct/total:.2f}%')
    return correct/total
# 测试函数
def test_accuracy(model, X_test, y_test, batch_size=64, device="cuda"):
    model.eval()
    model.to(device)
    correct = 0
    total = len(y_test)
    num_batches = (total + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min(total, start + batch_size)
            X_batch = X_test[start:end]
            y_batch = y_test[start:end]
            
            outputs = model(X_batch)  
            preds = torch.argmax(outputs, dim=1)  
            
            correct += (preds == y_batch).sum().item()  
            
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy*100:.2f}%')

# 8.训练
vocab_size = len(vocab)
embed_dim = 64
num_classes = 5
num_epochs = 10
# model = TextCNN(vocab_size,embed_dim,num_classes).to(device)
# loss = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-4)
# train_evaluate(model,optimizer,num_epochs,X_train,y_train,X_valid,y_valid,loss)

# 9.测试
# (1) 测试不同损失函数
# def mse_loss(y_pred,y_true):
#     batch_size,num_classes = y_pred.shape
#     labels = torch.zeros(batch_size, num_classes, device=y_pred.device)
#     for i in range(batch_size):
#         labels[i][y_true[i]] = 1.0
#     return torch.sum((y_pred - labels) ** 2) / batch_size

# def kl_divergence(y_pred, y_true):
#     log_p = F.log_softmax(y_pred, dim=1)  # log Q(x)
#     true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()
#     return F.kl_div(log_p, true_one_hot, reduction='batchmean')

# def confidence_penalty(y_pred, y_true, lambda_=0.1):
#     y_pred_prob = torch.clamp(F.softmax(y_pred, dim=1), 1e-12, 1.0)
#     ce = F.nll_loss(torch.log(y_pred_prob), y_true)
#     pred_classes = torch.argmax(y_pred_prob, dim=1)
#     wrong_mask = (pred_classes != y_true)
#     wrong_count = wrong_mask.sum().item()
#     batch_size = y_pred.shape[0]
#     confidence = y_pred_prob.max(dim=1).values
#     if wrong_count == 0:
#         penalty = torch.tensor(0.0, device=y_pred.device)
#     elif wrong_count / batch_size >= 0.1:
#         penalty = confidence[wrong_mask].sum() / batch_size
#     else:
#         penalty = confidence[wrong_mask].sum() / wrong_count
#     return ce + lambda_ * penalty

# def compare_losses(loss_dict, X_train, y_train, X_valid, y_valid, num_epochs=10, batch_size=64, device='cuda',):
#     acc_dict = {}
#     for loss_name, loss_fn in loss_dict.items():
#         print(f"Training with loss: {loss_name}")
#         model = TextCNN(vocab_size,embed_dim,num_classes).to(device) # 每次都要初始化模型和优化器！！
#         optimizer = optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-4)
#         acc = train_evaluate(model, optimizer, num_epochs, X_train, y_train, X_valid, y_valid,loss_fn, batch_size=batch_size, device=device)
#         acc_dict[loss_name] = acc*100
#     return acc_dict

# def plot_acc_dict(acc_dict):
#     loss_names = list(acc_dict.keys())
#     accuracies = list(acc_dict.values())
    
#     plt.figure(figsize=(10,6))
#     bars = plt.bar(loss_names, accuracies, color='skyblue')
    
#     plt.xlabel("损失函数")
#     plt.ylabel("准确率 (%)")
#     plt.title("不同损失函数对应的准确率比较")
#     plt.ylim(0, 100)
    
#     # 给每个柱子顶部加上百分比数字标签
#     for bar, acc in zip(bars, accuracies):
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{acc:.2f}%", ha='center', va='bottom', fontsize=10)
    
#     plt.show()

# loss_dict = {
#     'CrossEntropy': torch.nn.CrossEntropyLoss(),
#     'MSE': mse_loss,
#     'KL':kl_divergence,
#     'ConfidencePenalty':confidence_penalty
# }
# acc_dict = compare_losses(loss_dict,X_train,y_train,X_valid,y_valid)
# plot_acc_dict(acc_dict)

# (2) 测试不同学习率
# def compare_losses(X_train, y_train, X_valid, y_valid, num_epochs=10, batch_size=64, device='cuda'):
#     lr_list = [1e-5,1e-4,1e-3,1e-2,0.1]
#     acc_dict = {}
#     for lr in lr_list:
#         print(f"Training with learning rate: {lr}")
#         model = TextCNN(vocab_size,embed_dim,num_classes).to(device)
#         optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
#         acc = train_evaluate(model, optimizer, num_epochs, X_train, y_train, X_valid, y_valid, batch_size=batch_size, device=device)
#         acc_dict[lr] = acc*100
#     return acc_dict

# def plot_lr(acc_dict):
#     lrs = sorted(acc_dict.keys())
#     accuracies = [acc_dict[lr] for lr in lrs]
#     plt.figure(figsize=(8,5))
#     plt.plot(lrs, accuracies, marker='o', color='dodgerblue', linewidth=2)
#     plt.xscale('log')  # 学习率用对数刻度更直观
#     plt.xlabel('学习率 (log scale)', fontsize=12)
#     plt.ylabel('准确率 (%)', fontsize=12)
#     plt.title('不同学习率对应的准确率比较', fontsize=14)
#     plt.ylim(0, 100)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     for lr, acc in zip(lrs, accuracies):
#         plt.text(lr, acc + 1, f"{acc:.2f}%", ha='center', fontsize=10)
#     plt.show()

# acc_dict = compare_losses(X_train,y_train,X_valid,y_valid)
# plot_lr(acc_dict)


# (3) 测试不同卷积核个数
# def compare_losses(X_train, y_train, X_valid, y_valid, num_epochs=10, batch_size=64, device='cuda'):
#     embed_dim_list = [50,100,150,200,250]
#     acc_dict = {}
#     for embed_dim in embed_dim_list:
#         print(f"Training with embed_dim: {embed_dim} * 3")
#         model = TextCNN(vocab_size,embed_dim=embed_dim,num_classes=num_classes).to(device)
#         optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
#         acc = train_evaluate(model, optimizer, num_epochs, X_train, y_train, X_valid, y_valid, batch_size=batch_size, device=device)
#         acc_dict[embed_dim] = acc*100
#     return acc_dict

# def plot_embed_dim(acc_dict):
#     embed_dims = sorted(acc_dict.keys())
#     accuracies = [acc_dict[dim] for dim in embed_dims]
#     plt.figure(figsize=(8,5))
#     plt.plot(embed_dims, accuracies, marker='o', color='coral', linewidth=2)
#     plt.xlabel('Embedding 维度', fontsize=12)
#     plt.ylabel('准确率 (%)', fontsize=12)
#     plt.title('不同 Embedding 维度对准确率的影响', fontsize=14)
#     plt.ylim(0, 100)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     for dim, acc in zip(embed_dims, accuracies):
#         plt.text(dim, acc + 1, f"{acc:.2f}%", ha='center', fontsize=10)
#     plt.show()

# acc_dict = compare_losses(X_train,y_train,X_valid,y_valid)
# plot_embed_dim(acc_dict)

# (4) 测试不同优化器
# def compare_optimizers(X_train, y_train, X_valid, y_valid, num_epochs=10, batch_size=64, device='cuda'):
#     optimizer_dict = {
#         'SGD': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
#         'Adam': lambda params: torch.optim.Adam(params, lr=1e-3,weight_decay=1e-4),
#         'RMSprop': lambda params: torch.optim.RMSprop(params, lr=1e-3),
#         'Adagrad': lambda params: torch.optim.Adagrad(params, lr=1e-2),
#         'AdamW': lambda params: torch.optim.AdamW(params, lr=1e-3)
#     }
#     acc_dict = {}
#     for name, optimizer in optimizer_dict.items():
#         print(f"Training with optimizer: {name}")
#         model = TextCNN(vocab_size, embed_dim=128, num_classes=num_classes).to(device)
#         optimizer = optimizer(model.parameters())
#         acc = train_evaluate(model, optimizer, num_epochs,
#                              X_train, y_train, X_valid, y_valid,
#                              batch_size=batch_size, device=device)
#         acc_dict[name] = acc * 100  
#     return acc_dict

# def plot_optimizer(acc_dict):
#     names = list(acc_dict.keys())
#     values = [acc_dict[name] for name in names]
#     plt.figure(figsize=(8,5))
#     bars = plt.bar(names, values, color='slateblue')
#     plt.ylim(0, 100)
#     plt.ylabel('准确率 (%)', fontsize=12)
#     plt.title('不同优化器对准确率的影响', fontsize=14)
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     for bar, acc in zip(bars, values):
#         plt.text(bar.get_x() + bar.get_width()/2, acc + 1,
#                  f"{acc:.2f}%", ha='center', fontsize=10)
#     plt.show()

# acc_results = compare_optimizers(X_train, y_train, X_valid, y_valid)
# plot_optimizer(acc_results)


# (5) 随机初始化embedding vs glove
# def load_glove_embedding(glove_path, vocab, embed_dim):
#     embedding = torch.randn(len(vocab), embed_dim) * 0.1  # 默认随机初始化
#     word2idx = vocab.stoi if hasattr(vocab, 'stoi') else vocab  
#     with open(glove_path, 'r', encoding='utf8') as f:
#         for line in f:
#             tokens = line.strip().split()
#             word = tokens[0]
#             if word in word2idx:
#                 vec = torch.tensor([float(val) for val in tokens[1:]], dtype=torch.float)
#                 embedding[word2idx[word]] = vec
#     return embedding

# def compare_embedding_sources(X_train, y_train, X_valid, y_valid, vocab, glove_path,
#                                embed_dim=embed_dim, num_epochs=10, batch_size=64, device='cuda'):
#     acc_dict = {}
#     print("Training with random initialized embedding...")
#     model_random = TextCNN(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=num_classes).to(device)
#     optimizer = torch.optim.Adam(model_random.parameters(), lr=1e-3)
#     acc = train_evaluate(model_random, optimizer, num_epochs, X_train, y_train, X_valid, y_valid,
#                          batch_size=batch_size, device=device)
#     acc_dict["Random"] = acc * 100

#     print("Loading GloVe embeddings...")
#     glove_vectors = load_glove_embedding(glove_path, vocab, embed_dim)
#     model_glove = TextCNN(vocab_size=len(vocab), embed_dim=embed_dim, num_classes=num_classes).to(device)
#     model_glove.embedding.weight.data.copy_(glove_vectors)

#     print("Training with GloVe-initialized embedding...")
#     optimizer = torch.optim.Adam(model_glove.parameters(), lr=1e-3)
#     acc = train_evaluate(model_glove, optimizer, num_epochs, X_train, y_train, X_valid, y_valid,
#                          batch_size=batch_size, device=device)
#     acc_dict["GloVe"] = acc * 100

#     return acc_dict

# def plot_embedding_comparison(acc_dict):
#     labels = list(acc_dict.keys())
#     values = [acc_dict[k] for k in labels]
#     plt.figure(figsize=(6, 5))
#     bars = plt.bar(labels, values, color=['gray', 'green'])
#     plt.ylim(0, 100)
#     plt.title("GloVe vs Random Embedding", fontsize=14)
#     plt.ylabel("准确率 (%)", fontsize=12)
#     for bar, acc in zip(bars, values):
#         plt.text(bar.get_x() + bar.get_width()/2, acc + 1,
#                  f"{acc:.2f}%", ha='center', fontsize=11)
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.show()

# acc_dict = compare_embedding_sources(X_train, y_train, X_valid, y_valid, vocab, glove_path=glove_path)
# plot_embedding_comparison(acc_dict)

# (6)CNN vs RNN vs Transformer
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=1, dropout=0.5):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(x)  # h_n shape: (num_layers, batch_size, hidden_size)
        out = self.fc(h_n[-1]) # batch_size,hidden_size
        return out

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, num_layers=2, dropout=0.1):
        super(TextTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        x = self.transformer(x)  
        x = x.mean(dim=0)  # 对一个句子的词求平均，batch_size,embed_dim
        out = self.fc(x) # batch_size,num_classes
        return out

def compare_models(X_train, y_train, X_valid, y_valid, vocab_size, embed_dim, num_classes,
                   num_epochs=10, batch_size=64, device='cuda'):
    model_dict = {
        'CNN': TextCNN(vocab_size, embed_dim, num_classes),
        'RNN': TextRNN(vocab_size, embed_dim, hidden_size=128, num_classes=num_classes),
        'Transformer': TextTransformer(vocab_size, embed_dim, num_heads=4, num_classes=num_classes)
    }

    acc_dict = {}

    for name, model in model_dict.items():
        print(f"Training {name}...")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        acc = train_evaluate(model, optimizer, num_epochs, X_train, y_train, X_valid, y_valid,
                             loss_fc=torch.nn.CrossEntropyLoss(), batch_size=batch_size, device=device)
        acc_dict[name] = acc * 100

    return acc_dict

def plot_model_comparison(acc_dict):
    import matplotlib.pyplot as plt

    models = list(acc_dict.keys())
    accs = list(acc_dict.values())

    plt.figure(figsize=(8,5))
    bars = plt.bar(models, accs, color='cornflowerblue')
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom')
    plt.ylim(0, 100)
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Comparison: CNN vs RNN vs Transformer')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

acc_dict = compare_models(X_train, y_train, X_valid, y_valid, vocab_size, embed_dim, num_classes)
plot_model_comparison(acc_dict)
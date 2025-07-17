import os
import re
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
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
# 4.构建词汇表
def build_vocab(texts,ngram=1,max_vocab=10000):
    counter = Counter()
    for words in texts:
        if ngram == 1: # bow是1-gram
            counter.update(words)
        else:
            ngrams = [tuple(words[i:i+ngram]) for i in range(len(words) - ngram + 1)]
            counter.update(ngrams)
    contents = counter.most_common(max_vocab)
    vocab = {item:idx for idx,(item,_) in enumerate(contents)}
    return vocab

vocab = build_vocab(X_train)

# 5.文本转向量
def text_to_vector(texts, vocab, ngram=1):
    matrix = torch.zeros(len(texts), len(vocab))
    for i, words in enumerate(texts):
        if ngram == 1:
            word_counts = Counter(words)
        else:
            words = [tuple(words[j:j+ngram]) for j in range(len(words) - ngram + 1)]
            word_counts = Counter(words)

        for word, count in word_counts.items():  # 只看出现过的词
            if word in vocab:                   
                matrix[i][vocab[word]] = count
    return matrix

# bow
vocab_bow = build_vocab(X_train)
X_train_bow = text_to_vector(X_train,vocab_bow).to(device)
X_valid_bow = text_to_vector(X_valid,vocab_bow).to(device)
X_test_bow = text_to_vector(X_test,vocab_bow).to(device)
# n-gram
vocab_ngram = build_vocab(X_train,ngram=2)
X_train_ngram = text_to_vector(X_train,vocab_ngram,ngram=2).to(device)
X_valid_ngram = text_to_vector(X_valid,vocab_ngram,ngram=2).to(device)
X_test_ngram = text_to_vector(X_test,vocab_ngram,ngram=2).to(device)

# 6.定义模型
def relu(x):
    return x.clamp(min=0.0)
def softmax(x):
    exps = torch.exp(x-x.max(dim=1,keepdim=True).values)
    return exps / exps.sum(dim=1,keepdim=True)
class Mymodel(nn.Module):
    def __init__(self,input_dims=10000,hidden_dims=30,output_dims=5):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dims,hidden_dims) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dims))
        self.W2 = nn.Parameter(torch.randn(hidden_dims,output_dims) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(output_dims))
    def forward(self,x):
        layer1 = torch.matmul(x,self.W1) + self.b1
        layer1 = relu(layer1)
        layer2 = torch.matmul(layer1,self.W2) + self.b2
        output = softmax(layer2)
        return output # (batch_size,num_classes)
# 7.定义损失函数
# (1) CrossEntropy
def cross_entropy(y_pred,y_true):
    batch_size,num_classes = y_pred.shape
    labels = torch.zeros(len(y_true),num_classes,device=y_pred.device)
    for i in range(batch_size):
        num = y_true[i]
        labels[i][num] = 1
    return -torch.sum(labels * torch.log(y_pred + 1e-12)) / batch_size
# (2) MSE Loss
def mse_loss(y_pred,y_true):
    batch_size,num_classes = y_pred.shape
    labels = torch.zeros(batch_size, num_classes, device=y_pred.device)
    for i in range(batch_size):
        labels[i][y_true[i]] = 1.0
    return torch.sum((y_pred - labels) ** 2) / batch_size
# (3)KL散度
def kl_divergence(y_pred, y_true):
    batch_size, num_classes = y_pred.shape
    labels = torch.zeros(batch_size, num_classes, device=y_pred.device)
    for i in range(batch_size):
        labels[i][y_true[i]] = 1.0
    kl = labels * torch.log((labels + 1e-12) / (y_pred + 1e-12))
    return kl.sum() / batch_size

# (4) L = CrossEntropy + lambda*ConfidencePenalty
def confidence_penalty(y_pred,y_true,lambda_=0.1):
    batch_size,num_classes = y_pred.shape
    labels = torch.zeros(len(y_true),num_classes,device=y_pred.device)
    for i in range(batch_size):
        num = y_true[i]
        labels[i][num] = 1
    ce = -torch.sum(labels * torch.log(y_pred + 1e-12)) / batch_size
    pred_classes = torch.argmax(y_pred,dim=1)
    wrong_mask = (pred_classes != y_true) # 预测错的地方是True
    wrong_count = wrong_mask.sum().item()
    error_rate = wrong_count  / batch_size
    confidence = torch.max(y_pred,dim=1).values
    if wrong_count == 0:
        penalty = 0.0
    elif error_rate >= 0.1:
        penalty = confidence[wrong_mask].sum() / batch_size
    else:
        penalty = confidence[wrong_mask].sum() / wrong_count
    return ce + lambda_ * penalty

# 8.训练和评估函数
def train_evaluate(model,optimizer,epochs,X_train,y_train,X_valid,y_valid,loss_fc,batch_size=64,device="cuda"):
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
        print(f'epoch{epoch + 1} train Loss = {total_loss / num_batches},valid loss = {total_valid_loss / valid_batches},valid acc = {100*correct/total:.2f}%')
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
            
            outputs = model(X_batch)  # 预测概率分布
            preds = torch.argmax(outputs, dim=1)  # 选最大概率的类别
            
            correct += (preds == y_batch).sum().item()  # 统计正确数
            
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy*100:.2f}%')

# 9.开始训练
# model = Mymodel()
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
# num_epochs = 10
# batch_size = 64
# train_evaluate(model,optimizer,num_epochs,X_train_bow,y_train,X_valid_bow,y_valid,cross_entropy,batch_size)
# test_accuracy(model,X_test_bow,y_test)

# (1)比较bow和ngram
def compare_embed(X_train_bow, y_train, X_valid_bow, y_valid, X_train_ngram, X_valid_ngram, y_train_ngram, y_valid_ngram, epochs=10, batch_size=64, device='cuda'):
    results = {}
    print("Training BOW model...")
    model_bow = Mymodel(input_dims=X_train_bow.shape[1]).to(device)
    optimizer_bow = torch.optim.Adam(model_bow.parameters(), lr=0.001, weight_decay=1e-4)
    train_evaluate(model_bow, optimizer_bow, epochs, X_train_bow, y_train, X_valid_bow, y_valid, cross_entropy, batch_size, device)
    print("Testing BOW model...")
    test_accuracy(model_bow, X_valid_bow, y_valid, batch_size, device)

    model_bow.eval()
    with torch.no_grad():
        y_pred = model_bow(X_valid_bow)
        val_acc = (y_pred.argmax(dim=1) == y_valid).float().mean().item()
    results['BOW'] = val_acc

    print("\nTraining N-gram model...")
    model_ngram = Mymodel(input_dims=X_train_ngram.shape[1]).to(device)
    optimizer_ngram = torch.optim.Adam(model_ngram.parameters(), lr=0.001, weight_decay=1e-4)
    train_evaluate(model_ngram, optimizer_ngram, epochs, X_train_ngram, y_train_ngram, X_valid_ngram, y_valid_ngram, cross_entropy, batch_size, device)
    print("Testing N-gram model...")
    test_accuracy(model_ngram, X_valid_ngram, y_valid_ngram, batch_size, device)

    model_ngram.eval()
    with torch.no_grad():
        y_pred = model_ngram(X_valid_ngram)
        val_acc = (y_pred.argmax(dim=1) == y_valid_ngram).float().mean().item()
    results['N-gram'] = val_acc

    # 画图
    plt.bar(results.keys(), [v*100 for v in results.values()], color=['blue','orange'])
    plt.ylabel('Validation Accuracy (%)')
    plt.title('BOW vs N-gram Validation Accuracy')
    plt.show()
compare_embed(X_train_bow, y_train, X_valid_bow, y_valid, X_train_ngram, X_valid_ngram, y_train, y_valid, epochs=10, batch_size=64, device='cuda')

# (2)比较不同损失函数
def compare_loss_functions(X_train, y_train, X_valid, y_valid, vocab, epochs=10, batch_size=64, device='cuda'):
    losses = {
        'CrossEntropy': cross_entropy,
        'MSE': mse_loss,
        'KL Divergence': kl_divergence,
        'Confidence Penalty': confidence_penalty
    }
    results = {}
    for name, loss_fn in losses.items():
        print(f"Training with {name} loss...")
        model = Mymodel(input_dims=X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        train_evaluate(model, optimizer, epochs, X_train, y_train, X_valid, y_valid, loss_fn, batch_size, device)
        print(f"Testing model with {name} loss...")
        test_accuracy(model, X_valid, y_valid, batch_size, device)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_valid)
            val_acc = (y_pred.argmax(dim=1) == y_valid).float().mean().item()
        results[name] = val_acc

    plt.bar(results.keys(), [v*100 for v in results.values()], color=['blue','orange','green','red'])
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy for Different Loss Functions')
    plt.xticks(rotation=20)
    plt.show()

compare_loss_functions(X_train_bow, y_train, X_valid_bow, y_valid, vocab, epochs=10, batch_size=64, device='cuda')
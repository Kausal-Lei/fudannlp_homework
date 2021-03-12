import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from glove import Glove
from glove import Corpus
flag={"contradiction":0,"neutral":1,"entailment":2}
train=[]
test=[]
check=[]
for f in open('drive/snli_1.0_train.jsonl'):
    train.append(f)
for f in open('drive/snli_1.0_dev.jsonl'):
    check.append(f)
for f in open('drive/snli_1.0_test.jsonl'):
    test.append(f)
trainX=[]
trainY=[]
checkX=[]
checkY=[]
testX=[]
testY=[]
sentense = [["19260817"],["1000000007"]]
maxlen1=0
maxlen2=0
for i in range(len(train)):
    train[i]=json.loads(train[i])
    if flag.get(train[i]["gold_label"]) is None :
        continue
    a1 = train[i]["sentence1"].split()
    a2 = train[i]["sentence2"].split()
    sentense.append(a1)
    sentense.append(a2)
    maxlen1 = max(maxlen1, len(a1))
    maxlen2 = max(maxlen2, len(a2))
    trainX.append((a1, a2))
    trainY.append(flag[train[i]["gold_label"]])
for i in range(len(check)):
    check[i]=json.loads(check[i])
    if flag.get(check[i]["gold_label"]) is None :
        continue
    a1 = check[i]["sentence1"].split()
    a2 = check[i]["sentence2"].split()
    sentense.append(a1)
    sentense.append(a2)
    maxlen1 = max(maxlen1, len(a1))
    maxlen2 = max(maxlen2, len(a2))
    checkX.append((a1, a2))
    checkY.append(flag[check[i]["gold_label"]])
for i in range(len(test)):
    test[i] = json.loads(test[i])
    if flag.get(test[i]["gold_label"]) is None:
        continue
    a1 = test[i]["sentence1"].split()
    a2 = test[i]["sentence2"].split()
    sentense.append(a1)
    sentense.append(a2)
    maxlen1 = max(maxlen1, len(a1))
    maxlen2 = max(maxlen2, len(a2))
    testX.append((a1, a2))
    testY.append(flag[test[i]["gold_label"]])

corpus_model = Corpus()
corpus_model.fit(sentense, window=10)
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)

embedding_dim=300
batch_size = 32
type_size=3
dict_size=len(corpus_model.dictionary)
maxlen = maxlen1 + maxlen2
glove = Glove(no_components=embedding_dim, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=1,no_threads=8, verbose=True)
glove.add_dictionary(corpus_model.dictionary)
#glove.load('glove.model')
def get(X,Y,maxlen1,maxlen2):
    tmpX = np.zeros((len(X),maxlen1+maxlen2+1), dtype=int)
    for i in range(len(X)):
        a1 = X[i][0]
        a2 = X[i][1]
        while len(a1) < maxlen1:
            a1.append("19260817")
        while len(a2) < maxlen2:
            a2.append("19260817")
        a=a1
        a.append("1000000007")
        for j in a2:
            a.append(j)
        for j in range(len(a)):
            tmpX[i][j]=glove.dictionary[a[j]]
            #print(X[i][j])
            #print(tmpX[i][j])
    if Y is not None:
        tmpY = np.zeros(len(Y), dtype=int)
        for i in range(len(Y)):
            tmpY[i]=Y[i]
        return tmpX,tmpY
    return tmpX
class Sentence(Dataset):
    def __init__(self, x, y=None, maxlen1=10):
        #print(x.shape)
        x=torch.LongTensor(x).transpose(0, 1)
        self.x = (x[:maxlen1].transpose(0, 1), x[maxlen1:].transpose(0, 1))
        #print(self.x[0].shape)
        #print(self.x[1].shape)
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
trainX, trainY = get(X=trainX,Y=trainY,maxlen1=maxlen1,maxlen2=maxlen2)
checkX, checkY = get(X=checkX,Y=checkY,maxlen1=maxlen1,maxlen2=maxlen2)
testX, testY = get(X=testX,Y=testY,maxlen1=maxlen1,maxlen2=maxlen2)
train_set = Sentence(trainX, trainY,maxlen1)
check_set = Sentence(checkX, checkY,maxlen1)
test_set = Sentence(testX, testY,maxlen1)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # shuffle表示是否在每个epoch开始的时候，对数据进行重新排序
check_loader = DataLoader(check_set, batch_size=batch_size, shuffle=True)  # shuffle表示是否在每个epoch开始的时候，对数据进行重新排序
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

class ESIM(nn.Module):
    def __init__(self, dict_size, emb_dim, context_size,type_size,maxlen1,maxlen2):#字典大小，em维度，句子长度，类别大小
        #每一个句子被=[单词个数，em维度]
        super(ESIM, self).__init__()
        self.maxlen1 = maxlen1
        self.maxlen2 = maxlen2
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(dict_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.LongTensor(glove.word_vectors.tolist()))
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.emb_dim*8, self.emb_dim, batch_first=True, bidirectional=True)
        #print("shape",self.embedding.shape)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.emb_dim * 8),
            nn.Linear(self.emb_dim * 8, self.emb_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.emb_dim),
            nn.Dropout(0.5),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.emb_dim),
            nn.Dropout(0.5),
            nn.Linear(self.emb_dim, 3),
            nn.Softmax()
        )

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    # 编写前向过程
    def forward(self, inputs):
        #print(inputs[0][0].shape," ",self.maxlen1)
        # emb1: batch_size * seq_len1 * emb_dim
        emb1 = self.embedding1(inputs[0])
        # emb2: batch_size * seq_len2 * emb_dim
        emb2 = self.embedding2(inputs[1])
        # x1: batch_size * seq_len1 * emb_dim*2
        x1, _ = self.lstm(emb1)
        # x2: batch_size * seq_len2 * emb_dim*2
        x2, _ = self.lstm(emb2)

        # attention: batch_size * seq_len1 * seq_len2
        attention = torch.matmul(x1, x2.transpose(1, 2))

        # w1: batch_size * seq_len1 * seq_len2
        w1 = F.softmax(attention, dim=-1)
        # w2: batch_size * seq_len2 * seq_len1
        w2 = F.softmax(attention.transpose(1, 2), dim=-1)

        # X1 : batch_size * seq_len1 * emb_dim*2
        X1 = torch.matmul(w1, x2)
        # X2 : batch_size * seq_len2 * emb_dim*2
        X2 = torch.matmul(w2, x1)

        # mix1: batch_size * seq_len1 * (emb_dim*8)
        mix1 = torch.cat([x1, X1, self.submul(x1, X1)], -1)
        # mix2: batch_size * seq_len2 * (emb_dim*8)
        mix2 = torch.cat([x2, X2, self.submul(x2, X2)], -1)

        ans1,_ =  self.lstm2(mix1)
        ans2,_ =  self.lstm2(mix2)

        q1_rep = self.apply_multiple(ans1)
        q2_rep = self.apply_multiple(ans2)

        x = torch.cat([q1_rep, q2_rep], -1)
        return self.fc(x)

model = ESIM(dict_size,embedding_dim,maxlen1+maxlen2+1,type_size,maxlen1,maxlen2).cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
num_epoch = 100
for epoch in range(num_epoch):
    train_acc = 0.0
    check_acc = 0.0
    test_acc = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        #print(data[0])
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新参数
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(check_loader):
            check_pred = model(data[0].cuda())
            check_acc += np.sum(np.argmax(check_pred.cpu().data.numpy(), axis=1) == data[1].numpy())

        for i, data in enumerate(test_loader):
            test_pred = model(data[0].cuda())
            test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
    print(epoch, " ", train_acc/len(train_set)," ",check_acc/len(check_set)," ",test_acc/len(test_set))
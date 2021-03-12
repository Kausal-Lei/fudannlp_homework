
from __future__ import print_function
import argparse
from glove import Glove
from glove import Corpus
import pprint
#import gensim

sentense = []
for line in open("drive/sentense.txt"):
    a = line.split()
    a = [x.lower() for x in a if isinstance(x, str)]
    sentense.append(a)
corpus_model = Corpus()
corpus_model.fit(sentense, window=10)
#corpus_model.save('corpus.model')
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)
glove = Glove(no_components=300, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=20,
          no_threads=4, verbose=True)
glove.add_dictionary(corpus_model.dictionary)
glove.save('glove.model')
#print(glove.word_vectors[glove.dictionary["19260817"]].shape)
# 指定词条词向量


import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
dict={}
times={}
dim = 5
maxlen = 0
'''def get(X,Y,rate = 0.20):
    size = int(len(X) * (1 - rate))
    return size,len(X)-size,cg(X[:size]), cgy(Y[:size]), cg(X[size:]), cgy(Y[size:])'''
dataX=[]
dataY=[]
testdataX=[]
tot=1
for line in open("drive/train.tsv"): #处理train数据
    a = line.split()
    a = [x.lower() for x in a if isinstance(x, str)]
    a.remove(a[0])
    a.remove(a[0])
    #print(a[-1])
    dataY.append(int(a[-1]))
    a.remove(a[-1])
    for i in a:
        if dict.get(i) is None:
            dict[i] = tot
            tot += 1
    maxlen = max(maxlen,len(a))
    dataX.append(a)
for line in open("drive/test.tsv"): #处理test数据
    a = line.split()
    a = [x.lower() for x in a if isinstance(x, str)]
    a.remove(a[0])
    a.remove(a[0])
    for i in a :
        if dict.get(i) is None:
            dict[i] = tot
            tot += 1
    maxlen = max(maxlen, len(a))
    testdataX.append(a)
dict["19260817"]=0
#print(len(corpus_model.dictionary))
#print(dict.get("substantive"))
batch_size = 128
embedding_dim=300
type_size = 5
n_layers=1
hidden_size=64
dict_size=len(corpus_model.dictionary)

def get(X,Y,maxlen,embedding_dim):
    tmpX = np.zeros((len(X),maxlen), dtype=int)
    for i in range(len(X)):
        while len(X[i]) < maxlen:
            X[i].append("19260817")
        for j in range(len(X[i])):
            tmpX[i][j]=glove.dictionary[X[i][j]]
            #print(X[i][j])
            #print(tmpX[i][j])
    if Y is not None:
        tmpY = np.zeros(len(Y), dtype=int)
        for i in range(len(Y)):
            tmpY[i]=Y[i]
        return tmpX,tmpY
    return tmpX

class Sentence(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.LongTensor(x)
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
def spilt(set,rate):
    n = int((1-rate)*len(set))
    train_set=set[:n]
    check_set=set[n:]
    return train_set,check_set
dataX, data_checkX = spilt(dataX,rate=0.08)
dataY, data_checkY = spilt(dataY,rate=0.08)
trainX, trainY = get(X=dataX,Y=dataY,maxlen=maxlen,embedding_dim=embedding_dim)
checkX, checkY = get(X=data_checkX,Y=data_checkY,maxlen=maxlen,embedding_dim=embedding_dim)
testX = get(X=testdataX,Y=None,maxlen=maxlen,embedding_dim=embedding_dim)
train_set = Sentence(trainX, trainY)  # 把数据打包成[ [X], Y ]，X的形式为 句子个数*句子长度
check_set = Sentence(checkX, checkY)  # 把数据打包成[ [X], Y ]，X的形式为 句子个数*句子长度
test_set = Sentence(testX, y=None)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # shuffle表示是否在每个epoch开始的时候，对数据进行重新排序
check_loader = DataLoader(check_set, batch_size=batch_size, shuffle=True)  # shuffle表示是否在每个epoch开始的时候，对数据进行重新排序
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print("test",len(test_set))
#N,M,trainX,trainY,verifyX,verifyY = get(dataX,dataY)
class RNN(nn.Module):
    def __init__(self, dict_size, embedding_dim, context_size,type_size,n_layers,hidden_size):#字典大小，em维度，句子长度，类别大小,batch_size,rnn层数，每层RNN的神经元个数
        #每一个句子=[单词个数，em维度]
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(dict_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.LongTensor(glove.word_vectors.tolist()))
        self.n_directions = 2
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        #print("maxlen ", context_size, " ", self.fcn)
        '''
        input(seq_len, batch, input_size) 
        h0(num_layers * num_directions, batch, hidden_size) 
        c0(num_layers * num_directions, batch, hidden_size)

        输出数据格式： 
        output(seq_len, batch, hidden_size * num_directions) 
        hn(num_layers * num_directions, batch, hidden_size) 
        cn(num_layers * num_directions, batch, hidden_size)
        '''#一个隐藏层神经元的个数
        self.rnn = nn.Sequential (
            nn.LSTM(input_size=embedding_dim,\
            hidden_size=hidden_size, \
            num_layers=n_layers,\
            batch_first=True,\
            bidirectional=True),
        )
        self.fc = nn.Sequential (
            nn.Linear((hidden_size*self.n_directions) * context_size, 2*context_size),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(2*context_size, context_size),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(context_size, type_size),
            nn.Softmax(),
            nn.Dropout(p=0.5),
        )
    # 编写前向过程
    def forward(self, inputs):
        emb = self.embedding(inputs)
        out,hidden = self.rnn(emb)
        #print("out ",out.shape)
        #print(hh.shape,"       ",(self.hidden_size*self.n_directions) * self.embedding)
        log_probs = self.fc(out.reshape(out.size()[0], -1))
        return log_probs
#print("maxlen",maxlen)
def update(model,test_loader,data):
  model.eval()
  prediction = []
  with torch.no_grad():
      for i, data in enumerate(test_loader):
          #print(i)
          #print(data)
          test_pred = model(data.cuda())
          test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
          for y in test_label:
              prediction.append(y)
  with open("drive/predict.csv", 'w') as f:
      f.write('PhraseId,Sentiment\n')
      for i, y in  enumerate(prediction):
          f.write('{},{}\n'.format(i+156061, y))
model = RNN(dict_size,embedding_dim,maxlen,type_size,n_layers,hidden_size).cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0010)
num_epoch = 40
mx = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    check_acc = 0.0
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
    print(epoch, " ", train_acc/len(train_set),check_acc/len(check_set))
    if check_acc > mx:
      mx = check_acc
      update(model,test_loader,data)

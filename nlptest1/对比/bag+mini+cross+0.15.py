import sys
import random
import numpy as np
dict={}
times={}
dim = 5
def sigmoid(z) :
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
def ok(x) :
    for i in x:
        if i<'A' or (i>'Z' and i<'a') or i>'z':
            return False
    return True
def cg(X) : #X是个二维数组，第一维是行，我得表示为数据个数，第二维是列
    ans = np.zeros((len(X), len(dict)), dtype=int)
    id = 0
    for i in X:
        for j in i:
            if dict.get(j) is not None :
                ans[id][dict[j]] += 1
        id += 1
    return ans
def cgy(Y) :
    ans = np.zeros((len(Y), dim), dtype=int)
    id = 0
    for i in Y :
        ans[id][i] = 1
        id += 1
    return ans.T
def get(X,Y,rate = 0.01):
    size = int(len(X) * (1 - rate))
    return size,len(X)-size,cg(X[:size]), cgy(Y[:size]), cg(X[size:]), cgy(Y[size:])
dataX=[]
dataY=[]
testdataX=[]
tot=0
for line in open("drive/train.tsv"): #处理train数据
    a = line.split()
    a = [x.lower() for x in a if isinstance(x, str)]
    a.remove(a[0])
    a.remove(a[0])
    #print(a[-1])
    dataY.append(int(a[-1]))
    a.remove(a[-1])
    for i in a:
        if dict.get(i) is None and ok(i) == True:
            dict[i] = tot
            tot += 1
            times[i] = 1
        elif ok(i):
            times[i] += 1
    dataX.append(a)
for line in open("drive/test.tsv"): #处理test数据
    a = line.split()
    a = [x.lower() for x in a if isinstance(x, str)]
    a.remove(a[0])
    a.remove(a[0])
    for i in a :
        if dict.get(i) is None and ok(i) == True:
            dict[i] = tot
            tot += 1
            times[i] = 1
        elif ok(i):
            times[i] += 1
    testdataX.append(a)
for key,val in times.items() :
    if val <= 0 :
        del dict[key]
print(len(dict))
tot = 0
for key in dict:
    dict[key]=tot
    tot += 1
verifyX=[]
verifyY=[]
N,M,trainX,trainY,verifyX,verifyY = get(dataX,dataY)
#testX=cg(testdataX)
#print(N," ",M)
#print(trainX.shape[0]," ",trainX.shape[1])
print(trainY.shape[0]," ",trainY.shape[1])
#print(verifyX.shape[0]," ",verifyX.shape[1])
w = np.zeros((dim, len(dict)), dtype=float)
b = np.zeros(dim, dtype=float)
adagrad_w = np.zeros((dim, len(dict)), dtype=float)
adagrad_b = np.zeros(dim, dtype=float)
batch = 128
iter = 25
rate = 0.15
def predict(w,b,trainX) :
    z = np.matmul(trainX, w)
    z = z + b
    z = sigmoid(z)
    return z
def check(w,b,verifyX,verifyY,M,batch) :
    ans = np.zeros((dim, len(verifyX)), dtype=float)
    for d in range(dim):
        for idx in range(int(np.ceil(M/batch))):
            X = verifyX[idx * batch:min((idx + 1) * batch,M)]
            tmp = predict(w[d], b[d], X)
            limit=min((idx + 1) * batch,M)-idx * batch
            for i in range(limit) :
                ans[d][idx * batch+i]=tmp[i]
    ans=ans.T
    #print(ans.shape[0],ans.shape[1])
    sum = 0
    for i in range(len(verifyX)):
        idx = np.argmax(ans[i], axis=0)
        #print(idx)
        sum += (verifyY[idx][i] == 1)
    return 1.0*sum/(len(verifyX))
def cal (w,b,trainX,trainY) : #计算偏导更新
    z = predict(w,b,trainX)
    #print(trainY.shape[0]," ",z.shape[0])
    pred_error = trainY - z
    tmp = -np.sum(pred_error * trainX.T, 1)
    tmp2 = -(pred_error).sum()
    return tmp, tmp2
def shuffle (X,Y) :
  n = len(X)
  Y=Y.T
  for i in range(n) :
    id1=random.randrange(n)
    id2=random.randrange(n)
    X[id1],X[id2] = X[id2],X[id1]
    Y[id1], Y[id2] = Y[id2], Y[id1]
  return X,Y.T
for turn in range(iter):#枚举迭代次数
    #trainX,trainY  = shuffle(trainX,trainY )
    for idx in range(int(np.floor(N / batch))):#分批次更新
        for d in range(dim):#枚举维度   
            X = trainX[idx * batch:(idx + 1) * batch]
            Y = trainY[d][idx * batch:(idx + 1) * batch]
            w_grad, b_grad = cal(w[d], b[d], X, Y)
            #print(w_grad.shape[0])
            adagrad_w[d] += w_grad ** 2
            adagrad_b[d] += b_grad ** 2
            w[d] = w[d] - rate / (np.sqrt(adagrad_w[d] + 1e-8)) * w_grad
            b[d] = b[d] - rate / (np.sqrt(adagrad_b[d] + 1e-8)) * b_grad

    print("????? ",check(w,b,verifyX,verifyY,M,batch))
del trainX
del verifyX
import time
import csv
time.sleep(5)
def getans(w,b,testX,M,batch):
    ans = np.zeros((dim, len(testX)), dtype=float)
    for d in range(dim):
        for idx in range(int(np.ceil(M/batch))):
            X = testX[idx * batch:min((idx + 1) * batch,M)]
            tmp = predict(w[d], b[d], X)
            limit=min((idx + 1) * batch,M)-idx * batch
            for i in range(limit) :
                ans[d][idx * batch+i]=tmp[i]
    ans=ans.T
    #print(ans.shape[0],ans.shape[1])
    out = []
    for i in range(len(testX)):
        out.append(np.argmax(ans[i], axis=0))
    with open('drive/submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['PhraseId', 'Sentiment']
        # print(header)
        csv_writer.writerow(header)
        for i in range(len(out)):
            row = [str(i+156061), int(round(out[i]))]
            csv_writer.writerow(row)
            # print(row)
testX=cg(testdataX)
getans(w,b,testX,len(testX),batch)
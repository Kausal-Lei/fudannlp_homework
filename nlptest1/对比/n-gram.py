import sys
import random
import numpy as np
# import cupy as cp
import gc
import math
# gc.set_threshold(100000,10,10)
dict = [{}, {}, {}, {}, {}]
cnt = [0, 0, 0, 0, 0]
sum = [{}, {}, {}, {}, {}]
vis = {}
dim = 5


def ok(x):
    for i in x:
        if i < 'A' or (i > 'Z' and i < 'a') or i > 'z':
            return False
    return True


tot = 0
for line in open("./data/train.tsv"):  # 处理train数据
    a = line.split()
    #if vis.get(int(a[1])) is not None:
        #continue
    #print(a[0], " ", a[1])
    #vis[int(a[1])] = 1
    a = [x.lower() for x in a if isinstance(x, str)]
    a.remove(a[0])
    a.remove(a[0])
    # print(a[-1])
    d = int(a[-1])
    a.remove(a[-1])
    n = len(a)
    cnt[d] += n-1
    for i in range(n):
        if sum[d].get(a[i]) is None :
            sum[d][a[i]] = 0
        sum[d][a[i]] += 1
        if i == 0:
            continue
        if dict[d].get(str(a[i - 1] + " " + a[i])) is None:
            dict[d][str(a[i - 1] + " " + a[i])] = 1
        else:
            dict[d][str(a[i - 1] + " " + a[i])] += 1
k=1
out = []
for line in open("./data/test.tsv"):  # 处理test数据
    a = line.split()
    a = [x.lower() for x in a if isinstance(x, str)]
    #if vis.get(int(a[1])) is not None:
        #continue
    #vis[int(a[1])]=1
    #print(a[0], " ", a[1])
    a.remove(a[0])
    a.remove(a[0])
    rate=[0,0,0,0,0]
    for d in range(dim) :
        for i in range(len(a)) :
            if i == 0 :
                continue
            if sum[d].get(a[i-1]) is None:
                sum[d][a[i-1]] = 0
            s = str(a[i - 1] + " " + a[i])
            if dict[d].get(s) is None :
                dict[d][s]=0
            rate[d] += math.log(float(dict[d][s]+k)/(sum[d][a[i-1]]+k*20000*20000))
    #print(rate.index(max(rate)))
    out.append(rate.index(max(rate)))
import  csv
with open('submit1.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['PhraseId', 'Sentiment']
    # print(header)
    csv_writer.writerow(header)
    for i in range(len(out)):
        row = [str(i+156061), int(round(float(out[i])))]
        csv_writer.writerow(row)
        # print(row)
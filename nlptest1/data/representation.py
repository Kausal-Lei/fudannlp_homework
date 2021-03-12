import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import copy
from sklearn.model_selection import train_test_split


def sigmoid(x_data, w):
    return 1 / (1 + np.exp(-1 * np.dot(x_data.T, w)))


def logistic(x_data, y_data, w, word_num, w_num):
    iter_num = 100  # 循环次数
    sample_num = 2000  # SGD样本个数
    learning_rate = 1  # 学习率
    # m = x_data.shape[0]
    # print(x_data.shape)  # (18227, 156060)
    # print(y_data.shape)   # (156060, 1)
    # print(w.shape)  # (18227, 1)
    data = pd.DataFrame({
        'x': list(x_data),
        'y': list(y_data)
    })  # 便于用Dataframe.sample进行随机划分样本

    for i in range(iter_num):
        sgd_data = data.sample(n=sample_num)
        # print(sgd_data['x'])
        sgd_x = np.array(list(sgd_data['x'])).reshape(sample_num, word_num).T
        sgd_y = np.array(list(sgd_data['y'])).reshape(sample_num, 1)
        # print(sgd_x.shape)  # (18227, 500)
        # print(sgd_y.shape)  # (500, 1)
        cost_before = -1 * np.sum(
            sgd_y * np.log(sigmoid(sgd_x, w)) + (1 - sgd_y) * np.log(1 - sigmoid(sgd_x, w))) / sample_num
        delta = np.dot(sgd_x, sigmoid(sgd_x, w) - sgd_y) / sample_num
        w = w - learning_rate * delta
        cost_after = -1 * np.sum(
            sgd_y * np.log(sigmoid(sgd_x, w)) + (1 - sgd_y) * np.log(1 - sigmoid(sgd_x, w))) / sample_num
        print('%d-%d :%.4f ~ %.4f' % (w_num, i, cost_before, cost_after))

    return w


class BOW(object):

    def __init__(self, train_data, type_num):
        self.tdata = train_data  # 训练数据
        self.size = self.tdata.shape[0]  # 训练数据条数
        self.y_data = train_data['Sentiment']  # 训练数据的y值
        self.type_num = type_num  # 有几个分类

        # 统计单词个数并标记Sentiment
        self.tdata['Phrase'] = self.tdata['Phrase'].apply(lambda x: x.lower())  # 转换为小写
        temps = set()
        wid = dict()
        self.word_num = 0  # 不同单词（包括符号）个数
        for i in range(self.size):
            temp = self.tdata['Phrase'][i].split(' ')
            for j in range(len(temp)):
                if temp[j] not in temps:
                    wid[temp[j]] = self.word_num
                    self.word_num = self.word_num + 1
                    temps.add(temp[j])

        # print(word_num)  # 18227

        # 初始化可优化参数w和词向量vector
        np.random.seed(6)
        self.w = np.random.randn(self.word_num, self.type_num)
        # 生成BOW词向量
        print("kausal ", self.size, " ", self.word_num)
        self.vector = np.zeros((self.size, self.word_num), dtype=int)

        for i in range(self.size):
            temp = self.tdata['Phrase'][i].split(' ')
            for j in range(len(temp)):
                self.vector[i][wid[temp[j]]] += 1

        # 划分训练集和测试集，random_state让每次划分相同，test_size表示测试集的占比
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.vector, self.y_data,
                                                                                random_state=6, test_size=0.1)
        # print(len(self.x_test), self.x_test[0:5])

        # 对每个类别进行一对多的logistic回归进行训练
        for i in range(self.type_num):
            tempy = copy.deepcopy(self.y_train)
            tempy = np.array([1 if x == i else 0 for x in tempy]).reshape(len(self.y_train), 1)
            self.w[:, i] = list(logistic(self.x_train, tempy, self.w[:, i].reshape(self.word_num, 1), self.word_num, i))

    def match(self):
        x_test = np.array(self.x_test).reshape(len(self.x_test), self.word_num)
        temp_result = sigmoid(x_test.T, self.w)
        # print(temp_result.shape)  # (15606, 5)
        # print(np.max(temp_result, axis=1))
        result = np.argmax(temp_result, axis=1)  # 获取每行最大值的位置，即可能性最大的分类
        # print(result)
        matched = sum(result == self.y_test)

        print('acc = %.4f%%' % (matched / self.y_test.shape[0] * 100))
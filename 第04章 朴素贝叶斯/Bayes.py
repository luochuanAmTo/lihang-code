import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
#加载数据集
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names) #将数据集转换为 Pandas DataFrame
    df['label'] = iris.target   # 添加标签列
    df.columns = [    #重命名列名
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, :])   #取前 100 行数据（只包含两类），并转换为 NumPy 数组。
    # print(data)
    return data[:, :-1], data[:, -1]   #返回特征（前 4 列）和标签（最后一列）。

X, y = create_data()  #获取特征和标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)   #将数据集按 7:3 的比例划分为训练集和测试集

#朴素贝叶斯分类器类
class NaiveBayes:
    def __init__(self):
        self.model = None  #self.model 用于存储训练后的模型

    # 数学期望
    @staticmethod
    def mean(X):   #计算特征的均值
        return sum(X) / float(len(X))     #means = [5.0, 3.2, 1.4, 0.3]


    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))  #stdev(X) 将返回 2.828，即这个数据集的标准差

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train   计算特征的均值和标准差
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):  #训练模型
        labels = list(set(y))  #获取所有类别标签
        data = {label: [] for label in labels}   #初始化字典，按类别存储数据
        for f, label in zip(X, y):   # 将数据按类别分组
            data[label].append(f)
        self.model = {  #计算每个类别的特征均值和标准差
            label: self.summarize(value)
            for label, value in data.items()
        }
        return 'gaussianNB train done!'

    # 计算类别概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)    #计算联合概率。
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(
            self.calculate_probabilities(X_test).items(), #预测输入数据的类别。
            key=lambda x: x[-1])[-1][0]
        return label
  #计算模型在测试集上的准确率
    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):  #遍历测试集
            label = self.predict(X)
            if label == y:    # 统计正确分类数
                right += 1

        return right / float(len(X_test))   #返回准确率

#测试模型
model = NaiveBayes()  #创建对象
model.fit(X_train, y_train)  #训练模型
print(model.predict([4.4,  3.2,  1.3,  0.2]))  #预测新数据所属类别
model.score(X_test, y_test)  #计算准确率

#使用 Scikit-learn 的 GaussianNB  scikitlearn实例
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.predict([[4.4,  3.2,  1.3,  0.2]])

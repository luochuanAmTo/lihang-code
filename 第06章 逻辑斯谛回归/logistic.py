from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    # print(data)
    return data[:,:2], data[:,-1]   #该函数加载鸢尾花数据集，提取前 100 行的 sepal length、sepal width 和 label，并将其拆分为特征和标签。

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class LogisticReressionClassifier:  #逻辑回归分类器
    def __init__(self, max_iter=200, learning_rate=0.01):  #max_iter：最大迭代次数，默认值为 200  learning_rate：学习率，默认值为 0.01。
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))    #定义 Sigmoid 函数，将输入映射到 (0, 1) 区间。

    def data_matrix(self, X):  #定义 Sigmoid 函数，将输入映射到 (0, 1) 区间。
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat   #输入 X = [[1, 2], [3, 4]]，输出 data_mat = [[1.0, 1, 2], [1.0, 3, 4]]

    def fit(self, X, y):  #训练逻辑回归模型，更新权重
        # label = np.mat(y)
        data_mat = self.data_matrix(X)  # m*n   #X：训练集特征数据，形状为 (n_samples, n_features)
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose(
                    [data_mat[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
            self.learning_rate, self.max_iter))

    # def f(self, x):
    #     return -(self.weights[0] + self.weights[1] * x) / self.weights[2]

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)
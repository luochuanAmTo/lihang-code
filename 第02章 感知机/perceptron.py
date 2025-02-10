import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import Perceptron

iris = load_iris()    #加载 Iris 数据集。
print(iris)
print('='*50)
df = pd.DataFrame(iris.data, columns=iris.feature_names)  #将特征数据转换为 Pandas DataFrame，并设置列名。
print(df)
print('='*50)
df['label'] = iris.target   #将标签数据添加到 DataFrame 中。
print(df)
print('='*50)

df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]
print(df.label.value_counts()  ) #统计标签列中每个类别的样本数量。

print('='*50)

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0') #绘制前 50 个样本的花萼长度和花萼宽度的散点图，并标记为类别 0。
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')  #绘制第 50 到 100 个样本的花萼长度和花萼宽度的散点图，并标记为类别 1
plt.xlabel('sepal length')   #设置 x 轴的标签为 sepal length（花萼长度）。
plt.ylabel('sepal width')   #设置 y 轴的标签为 sepal width（花萼宽度）。
#plt.show()  #显示图形

data = np.array(df.iloc[:100, [0, 1, -1]])  #将前 100 行的花萼长度、花萼宽度和标签列提取出来，并转换为 NumPy 数组。  [0, 1, -1] 表示选择第 0 列、第 1 列和最后一列：
print(data)
print('='*50)

data = np.array(df.iloc[:100, [0, 1, -1]])  #将前 100 行的花萼长度、花萼宽度和标签列提取出来，并转换为 NumPy 数组。  [0, 1, -1] 表示选择第 0 列、第 1 列和最后一列：
X, y = data[:,:-1], data[:,-1]  #data[:, :-1]提取 data 数组中除最后一列之外的所有列。data[:, -1] 提取 data 数组中的最后一列。
y = np.array([1 if i == 1 else -1 for i in y])  #统一格式：将标签转换为 -1 和 1 可以简化感知机算法的实现

# 数据线性可分，二分类数据
# 此处为一元一次线性方程
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        #创建一个数组，长度为特征数量    self.w 是模型的权重向量，初始值为全 1。
        self.b = 0
        self.l_rate = 0.1  #学习率
        # self.data = data

    def sign(self, x, w, b):   #计算感知机预测值 y=w*x+b
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降法
    def fit(self, X_train, y_train): #X_train 是训练数据的特征矩阵，形状为 (n_samples, n_features)。


        is_wrong = False#is_wrong 初始值为 False，表示模型尚未完全正确分类所有样
        while not is_wrong:
            wrong_count = 0     #wrong_count 用于记录当前迭代中错误分类的样本数量。
            for d in range(len(X_train)):
                X = X_train[d]   #X_train[d] 是第 d 个样本的特征向量
                y = y_train[d] #y_train[d] 是第 d 个样本的标签
                if y * self.sign(X, self.w, self.b) <= 0:   #判断当前样本是否被错误分类。
                    self.w = self.w + self.l_rate * np.dot(y, X)    #更新权重向量 w
                    self.b = self.b + self.l_rate * y  #更新偏置项 b
                    wrong_count += 1   #增加错误分类样本的计数器。
            if wrong_count == 0:
                is_wrong = True    #当所有样本被正确分类时，设置 is_wrong 为 True，结束训练。
        return 'Perceptron Model!'

    def score(self):
        pass


perceptron = Model()
perceptron.fit(X, y)
print(perceptron.w)
print('='*50)
x_points = np.linspace(4, 7, 10)
print(x_points)
print('='*50)

y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]    #w0*x0+w1*x1+b=0
plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')   #绘制样本点
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

clf = Perceptron(fit_intercept=True,   #clf 是创建的感知机模型实例。
                 max_iter=1000,     #max_iter=1000：设置感知机训练时的最大迭代次数。即最多进行1000次更新
                 shuffle=True)      #每次训练时会对数据进行随机打乱，这样有助于提高模型的泛化能力
clf.fit(X, y)     #fit 是训练模型的函数

print(clf.coef_)    # Weights assigned to the features.
print(clf.intercept_)   # 截距

print('='*50)
# 画布大小
plt.figure(figsize=(10,10))

# 中文标题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('鸢尾花线性数据示例')

plt.scatter(data[:50, 0], data[:50, 1], c='b', label='Iris-setosa',)
plt.scatter(data[50:100, 0], data[50:100, 1], c='orange', label='Iris-versicolor')

# 画感知机的线
x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_points + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_points, y_)

# 其他部分
plt.legend()  # 显示图例
plt.grid(False)  # 不显示网格
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
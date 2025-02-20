import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()
print(iris)
print('='*60)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)
print('='*60)
df['label'] = iris.target
print(df)
print('='*60)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df)
print('='*60)

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')    #前五十行数据
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')  #50-100行数据
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('X'*50)
print(X_train)
print('X'*50)
print(y_train)
print()
class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):  #根据输入的特征数据 X 预测类别标签。
        # 取出n个点
        knn_list = []    #初始化一个空列表 knn_list，用于存储每个点与训练集中点的距离及其对应的标签。
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)  # 用于存储测试样本 X 与训练集中前 n 个点的距离及其标签
            knn_list.append((dist, self.y_train[i])) #计算出的距离和相应的标签 y_train[i] 以元组 (dist, label) 的形式存储到 knn_list 中。
        print(knn_list)   #?

        for i in range(self.n, len(self.X_train)):

            max_index = knn_list.index(max(knn_list, key=lambda x: x[0])) #找到 knn_list 中距离最大的点（max_index）。
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)      #计算测试样本 X 与当前训练样本 X_train[i] 的距离。
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])     #如果当前距离小于 knn_list 中的最大距离，则替换 knn_list 中的对应点。

        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn) #提取 knn_list 中所有点的标签。
#         max_count = sorted(count_pairs, key=lambda x: x)[-1]
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]  #使用 Counter 统计每个标签的频数。
        return max_count   #返回频数最高的标签。

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test): #遍历测试集 X_test 和 y_test。
            label = self.predict(X)
            if label == y:
                right_count += 1   #如果预测标签与真实标签 y 一致，则增加正确计数 right_count。
        return right_count / len(X_test)  # 返回正确预测的比例（准确率）。

clf = KNN(X_train, y_train)
c=clf.score(X_test, y_test)

test_point = [6.0, 3.0]
print('Test Point: {}'.format(clf.predict(test_point)))

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')  #测试样本
plt.xlabel('sepal length')   #特征一
plt.ylabel('sepal width')    #特征二
plt.show()


clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
clf_sk.score(X_test, y_test)

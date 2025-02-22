import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = np.array([[5, 12, 1], [6, 21, 0], [14, 5, 0], [16, 10, 0], [13, 19, 0],
                 [13, 32, 1], [17, 27, 1], [18, 24, 1], [20, 20,
                                                         0], [23, 14, 1],
                 [23, 25, 1], [23, 31, 1], [26, 8, 0], [30, 17, 1],
                 [30, 26, 1], [34, 8, 0], [34, 19, 1], [37, 28, 1]])
X_train = data[:, 0:2]   #提取前两列作为特征
y_train = data[:, 2]     #提取最后一列作为标签
#定义两个 K 近邻分类器
models = (KNeighborsClassifier(n_neighbors=1, n_jobs=-1),   #n_neighbors=1: 使用 1 个最近邻
          KNeighborsClassifier(n_neighbors=2, n_jobs=-1))    #n_neighbors=2: 使用 2 个最近邻
models = (clf.fit(X_train, y_train) for clf in models)  #使用生成器表达式对每个模型调用 fit 方法，用训练数据 X_train 和 y_train 训练模型

titles = ('K Neighbors with k=1', 'K Neighbors with k=2')  #k=1  he  k=2的模型

fig = plt.figure(figsize=(15, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_train[:, 0], X_train[:, 1]   #提取第0列和第1列

x_min, x_max = X0.min() - 1, X0.max() + 1  #定义 x 轴的范围（最小值减 1，最大值加 1）
y_min, y_max = X1.min() - 1, X1.max() + 1  #定义 y 轴的范围（最小值减 1，最大值加 1）
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

for clf, title, ax in zip(models, titles, fig.subplots(1, 2).flatten()):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Z))])
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    ax.scatter(X0, X1, c=y_train, s=50, edgecolors='k', cmap=cmap, alpha=0.5)
    ax.set_title(title)

plt.show()
from collections import namedtuple
import numpy as np


# 建立节点类
class Node(namedtuple("Node", "location left_child right_child")):
    def __repr__(self):              #Node: 使用 namedtuple 定义一个节点类，包含以下字段：
        return str(tuple(self))      # location: 当前节点的坐标。
                                       # left_child: 左子节点。
                                       # right_child: 右子节点。
# kd tree类
class KdTree():
    def __init__(self, k=1):
        self.k = k   #  k表示特征维度
        self.kdtree = None  #kdtree 用于存储构建的 KD 树

    # 构建kd tree
    def _fit(self, X, depth=0):
        try:
            k = self.k
        except IndexError as e:
            return None
        # 这里可以展开，通过方差选择axis
        axis = depth % k     #根据当前深度选择划分轴。
        X = X[X[:, axis].argsort()]   #按当前轴的值对数据进行排序
        median = X.shape[0] // 2  #找到中位数索引
        try:
            X[median]
        except IndexError:
            return None
        return Node(location=X[median],       #创建当前节点，并递归构建左右子树
                    left_child=self._fit(X[:median], depth + 1),
                    right_child=self._fit(X[median + 1:], depth + 1))

    def _search(self, point, tree=None, depth=0, best=None): #递归搜索最近邻。
        if tree is None:
            return best   # 如果当前节点为空，返回当前最佳节点。
        k = self.k
        # 更新 branch
        if point[0][depth % k] < tree.location[depth % k]:
            next_branch = tree.left_child   #根据当前轴的值决定搜索左子树还是右子树。
        else:
            next_branch = tree.right_child
        if not next_branch is None:
            best = next_branch.location  #更新最佳节点
        return self._search(point,   #递归调用 _search，继续搜索子树
                            tree=next_branch,
                            depth=depth + 1,
                            best=best)
#拟合 KD 树
    def fit(self, X):
        self.kdtree = self._fit(X)#调用 _fit 方法构建 KD 树，并返回根节点
        return self.kdtree

    def predict(self, X):   #predict: 调用 _search 方法查找最近邻，并返回结果
        res = self._search(X, self.kdtree)
        return res
#测试
KNN = KdTree()  #KNN = KdTree(): 创建 KD 树对象。
X_train = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]) #X_train: 训练数据集。
KNN.fit(X_train)  #构建KD树
X_new = np.array([[3, 4.5]]) #新数据点
res = KNN.predict(X_new)    #查找最近临

x1 = res[0]
x2 = res[1]

print("x点的最近邻点是({0}, {1})".format(x1, x2))
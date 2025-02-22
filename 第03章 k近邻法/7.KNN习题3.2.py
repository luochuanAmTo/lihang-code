import numpy as np
from sklearn.neighbors import KDTree

train_data = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])  #train_data 是一个二维数组，表示训练数据集。每行是一个样本点，包含两个特征值（例如坐标）
tree = KDTree(train_data, leaf_size=2)  #使用 train_data 构建 KD 树。
                                        #leaf_size=2: 设置叶子节点的最大样本数为 2，用于控制树的构建方式
dist, ind = tree.query(np.array([(3, 4.5)]), k=1)  #查询点 (3, 4.5) 的最近邻。
#k=1: 查找 1 个最近邻。
# dist: 返回最近邻的距离。
# ind: 返回最近邻在 train_data 中的索引。




x1 = train_data[ind[0]][0][0]  #ind[0]: 获取最近邻的索引。 x1: 最近邻的第一个特征值（例如 x 坐标
x2 = train_data[ind[0]][0][1]

print("x点的最近邻点是({0}, {1})".format(x1, x2))
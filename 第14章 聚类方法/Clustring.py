import math
import random
import numpy as np
from sklearn import datasets,cluster
import matplotlib.pyplot as plt

iris = datasets.load_iris()
gt = iris['target'];gt  #存储了数据集的标签。

iris['data'][:,:2].shape # 前两个特征
data = iris['data'][:,:2] #

x = data[:,0]   #x,y 分别取的一个和第二个特征
y = data[:,1]

plt.scatter(x, y, color='green')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()

# 定义聚类数的节点

class ClusterNode: #层次聚类的节点
    def __init__(self, vec, left=None, right=None, distance=-1, id=None, count=1):
        """
        :param vec: 保存两个数据聚类后形成新的中心
        :param left: 左节点
        :param right:  右节点
        :param distance: 两个节点的距离
        :param id: 用来标记哪些节点是计算过的
        :param count: 这个节点的叶子节点个数
        """
        self.vec = vec  #是该节点的向量表示，通常是两个子节点聚类后形成的新中心
        self.left = left #是该节点的左右子节点
        self.right = right
        self.distance = distance #是该节点与子节点之间的距离
        self.id = id #于标记节点，避免重复计算。
        self.count = count  #表示该节点的叶子节点个数，用于计算聚类时的权重。

def euler_distance(point1: np.ndarray, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


# 层次聚类（聚合法）
#自下而上
class Hierarchical:
    def __init__(self, k):
        self.k = k #用户指定的聚类数量，即最终希望将数据聚成多少类
        self.labels = None #用于存储每个数据点的聚类标签，初始值为 None。

    def fit(self, x): #方法是层次聚类的主要逻辑，负责将数据点逐步合并，直到剩下 k 个簇。
        nodes = [ClusterNode(vec=v, id=i) for i, v in enumerate(x)]  #将每个数据点初始化为一个 ClusterNode 对象，id 是数据点的索引，vec 是数据点的特征向量。
        distances = {}#用于缓存两个节点之间的距离，避免重复计算。
        point_num, feature_num = x.shape #数据点的数量和特征的数量
        self.labels = [-1] * point_num #初始化标签列表，值为 -1，表示尚未分配聚类标签
        currentclustid = -1
        while (len(nodes)) > self.k:
            min_dist = math.inf #记录当前最小的簇间距离，初始值为无穷大
            nodes_len = len(nodes)
            closest_part = None#记录距离最近的两个簇的索引。
            for i in range(nodes_len - 1) :  #双重循环遍历所有簇对，计算它们之间的距离：
                for j in range(i + 1, nodes_len):
                    d_key = (nodes[i].id, nodes[j].id)
                    if d_key not in distances: #如果距离没有被缓存过，则调用 euler_distance 计算欧拉距离，并缓存到 distances 中。
                        distances[d_key] = euler_distance(nodes[i].vec, nodes[j].vec)
                    d = distances[d_key]
                    if d < min_dist: #如果当前距离小于 min_dist，则更新 min_dist 和 closest_part。
                        min_dist = d
                        closest_part = (i, j)

            part1, part2 = closest_part #part1 和 part2 是最近的两个簇的索引。
            node1, node2 = nodes[part1], nodes[part2]  #node1 和 node2 是这两个簇对应的节点。
            new_vec = [(node1.vec[i] * node1.count + node2.vec[i ] * node2.count) / (node1.count + node2.count)#计算新簇的中心向量，
                       # 采用加权平均的方式（权重为每个簇的叶子节点数量 count）。
                       for i in range(feature_num)]
            new_node = ClusterNode(vec=new_vec,    #创建一个新的 ClusterNode，表示合并后的簇，其左右子节点为 node1 和 node2。
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   id=currentclustid,
                                   count=node1.count + node2.count)
            currentclustid -= 1 #递减，为新簇分配唯一的 id
            del nodes[part2], nodes[part1]
            nodes.append(new_node) #删除原来的两个簇 node1 和 node2，并将新簇 new_node 添加到 nodes 中。

        self.nodes = nodes
        self.calc_label()

    def calc_label(self):
        """
        调取聚类的结果
        """
        for i, node in enumerate(self.nodes):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        """
        递归遍历叶子节点
        """
        if node.left == None and node.right == None:
            self.labels[node.id] = label  #如果当前节点是叶子节点（left 和 right 都为 None），则将其 id 对应的标签设置为 label。
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)

#测试：
my = Hierarchical(3)
my.fit(data)
labels = np.array(my.labels)
print(labels)



cat1 = data[np.where(labels==0)]
cat2 = data[np.where(labels==1)]
cat3 = data[np.where(labels==2)]

plt.scatter(cat1[:,0], cat1[:,1], color='green')
plt.scatter(cat2[:,0], cat2[:,1], color='red')
plt.scatter(cat3[:,0], cat3[:,1], color='blue')
plt.title('Hierarchical clustering with k=3')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()
#调用cluster方法
sk = cluster.AgglomerativeClustering(3)
sk.fit(data)
labels_ = sk.labels_
print(labels_)


# kmeans

class MyKmeans:
    def __init__(self, k, n=20):
        self.k = k#用户指定的聚类数量，即希望将数据划分为多少个簇。
        self.n = n  #：最大迭代次数，默认值为 20，表示算法最多运行 20 次迭代。

    def fit(self, x, centers=None):
        # 第一步，随机选择 K 个点, 或者指定  随机选择 k 个数据点作为初始聚类中心。
        if centers is None:
            idx = np.random.randint(low=0, high=len(x), size=self.k) #从数据集中随机选择 k 个索引。
            centers = x[idx]   #根据索引从数据集中提取对应的点作为初始聚类中心。
        # print(centers)

        inters = 0
        while inters < self.n:
            # print(inters)
            # print(centers)
            points_set = {key: [] for key in range(self.k)}
            #初始化点集 points_set：是一个字典，键是簇的索引（0 到 k-1），值是属于该簇的数据点列表。
            # 第二步，遍历所有点 P，将 P 放入最近的聚类中心的集合中
            for p in x: #分配数据点到最近的簇
                nearest_index = np.argmin(np.sum((centers - p) ** 2, axis=1) ** 0.5)
                points_set[nearest_index].append(p)

            # 第三步，遍历每一个点集，计算新的聚类中心
            for i_k in range(self.k):
                centers[i_k] = sum(points_set[i_k]) / len(points_set[i_k])

            inters += 1

        return points_set, centers
# points_set：是一个字典，键是簇的索引，值是属于该簇的数据点列表。
#
# centers：是最终的聚类中心。

m = MyKmeans(3)
points_set, centers = m.fit(data)

# visualize result

cat1_ = np.asarray(points_set[0])
cat2_ = np.asarray(points_set[1])
cat3_ = np.asarray(points_set[2])

for ix, p in enumerate(centers):
    plt.scatter(p[0], p[1], color='C{}'.format(ix), marker='^', edgecolor='black', s=256)

plt.scatter(cat1_[:, 0], cat1_[:, 1], color='green')
plt.scatter(cat2_[:, 0], cat2_[:, 1], color='red')
plt.scatter(cat3_[:, 0], cat3_[:, 1], color='blue')
plt.title('Hierarchical clustering with k=3')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()

#使用库
# using sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, max_iter=100).fit(data)
gt_labels__ = kmeans.labels_
centers__ = kmeans.cluster_centers_
# gt_labels__
# centers__


 #绘制不同 K 值对应的损失值（即簇内误差平方和，Inertia）来观察损失值的变化趋势，从而选择一个合适的 K 值。
from sklearn.cluster import KMeans

loss = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=100).fit(data)
    loss.append(kmeans.inertia_ / len(data) / 3)  #K-Means 模型的簇内误差平方和（Inertia），表示所有样本到其所属簇中心的距离平方和。

plt.title('K with loss')
plt.plot(range(1, 10), loss)
plt.show()
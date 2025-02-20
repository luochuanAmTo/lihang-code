from math import sqrt
from collections import namedtuple

# kd-tree每个结点中主要包含的数据结构如下
class KdNode(object):  #kd树中的一个节点
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）   当前节点用于分割的维度序号（例如，0 表示第 1 维，1 表示第 2 维，依此类推）。
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree


class KdTree(object):
    def __init__(self, data):
        k = len(data[0])  # 数据维度

        def CreateNode(split, data_set):  # 按第split维划分数据集exset创建KdNode split：当前分割的维度序号 data_set：当前需要处理的数
            if not data_set:  # 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            #data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])  #对数据集按照当前维度进行排序
            split_pos = len(data_set) // 2  # //为Python中的整数除法  找到中位数
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # cycle coordinates 计算下一个分割维度的序号，通过取模实现维度循环

            # 递归的创建kd树
            return KdNode(
                median,  #：当前节点的数据点
                split,   #当前节点的分割维度。
                CreateNode(split_next, data_set[:split_pos]),  # 递归创建左子树，使用分割点左侧的数据集。
                CreateNode(split_next, data_set[split_pos + 1:]))  # 创建右子树，使用分割点右侧的数据集。

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)   #打印当前节点
    if root.left:  # 节点不为空
        preorder(root.left)  #递归遍历左子树。
    if root.right:
        preorder(root.right)  #递归遍历右子树


# 定义一个namedtuple,分别存放最近坐标点、最近邻点与目标点的距离和访问过的节点数
result = namedtuple("Result_tuple",
                    "nearest_point  nearest_dist  nodes_visited")


def find_nearest(tree, point):  #tree：构建好的kd树对象。
    k = len(point)  # 数据维度    point：目标点（待查询最近邻的点）

    def travel(kd_node, target, max_dist):  #kd_node：当前处理的kd树节点。max_dist：当前允许的最大搜索距离
        if kd_node is None:  #终止条件：如果当前节点为空，返回一个初始结果（无效点、无穷大距离、0访问节点）
            return result([0] * k, float("inf"),
                          0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1 #当前节点计数为1。

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴” 当前节点保存的数据点（分割超平面通过此点）。

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        #----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归
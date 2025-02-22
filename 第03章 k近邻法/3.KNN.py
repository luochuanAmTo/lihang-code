import math
from itertools import combinations
def L(x, y, p=2):         #计算欧式距离
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0


x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]
for i in range(1, 5):
    r = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
    print(min(zip(r.values(), r.keys())))     #  计算 x1 与 x2 和 x3 之间的距离（p 从 1 到 4）找到每个 p 值下的最小距离及其对应的向量对。





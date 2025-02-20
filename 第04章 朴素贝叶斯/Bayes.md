## 朴素贝叶斯的条件独立性假设

  

朴素贝叶斯的核心假设是：**在给定类别 Y=ck\*Y\*=\*c\**k\* 的条件下，所有特征 X(1),X(2),…,X(n)\相互独立**。这意味着，每个特征 X(j)的出现与否只依赖于类别 Y，而与其他特征无关。
$$
P(X = x \mid Y = c_k) = P(X^{(1)} = x^{(1)}, X^{(2)} = x^{(2)}, \dots, X^{(n)} = x^{(n)} \mid Y = c_k)
$$

![image-20250219214310622](C:\Users\AmTo2\AppData\Roaming\Typora\typora-user-images\image-20250219214310622.png)

![image-20250219214907566](C:\Users\AmTo2\AppData\Roaming\Typora\typora-user-images\image-20250219214907566.png)

这是一个典型的分类问题，**转为数学问题就是比较p(嫁|(不帅、性格不好、身高矮、不上进))与p(不嫁|(不帅、性格不好、身高矮、不上进))的概率**，谁的概率大，我就能给出嫁或者不嫁的答案！

![image-20250219214926852](C:\Users\AmTo2\AppData\Roaming\Typora\typora-user-images\image-20250219214926852.png)

![image-20250219215204164](C:\Users\AmTo2\AppData\Roaming\Typora\typora-user-images\image-20250219215204164.png)

import torch as t

# 返回一个符合均值为0，方差为1的正态分布（标准正态分布）中填充随机数的张量
print(t.randn(4))
print(t.randn(1,4))
print(t.randn(2,2,3))
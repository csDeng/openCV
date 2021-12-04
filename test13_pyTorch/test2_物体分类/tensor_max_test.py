import torch

a = torch.tensor([[1,5,4,3],[2,4,3,1],[77,66,99,55]])

print(a)

'''
使用max()函数对softmax函数的输出值进行操作，求出预测值索引，然后与标签进行比对，计算准确率
input是softmax函数输出的一个tensor
dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值

=> 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
'''
print(torch.max(a,1))


'''
tensor([[ 1,  5,  4,  3],
        [ 2,  4,  3,  1],
        [77, 66, 99, 55]])
torch.return_types.max(
values=tensor([ 5,  4, 99]),
indices=tensor([1, 1, 2]))
'''

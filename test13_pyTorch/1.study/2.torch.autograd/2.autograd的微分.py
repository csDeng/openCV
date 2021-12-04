import torch

# @目标：训练数据计算梯度（导数）


a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True )

Q = 3*a**3 - b**2
# print(Q)    # tensor([-12.,  65.], grad_fn=<SubBackward0>)
'''
假设a和b是神经网络的参数，Q是误差。
在 NN 训练中，我们想要相对于参数的误差
'''

externel_grad = torch.tensor([1., 1.])
Q.backward(gradient=externel_grad)


print(9*a**2 == a.grad)
print(-2*b == b.grad)
# tensor([True, True])
# tensor([True, True])

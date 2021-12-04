import torch, torchvision

'''
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
'''

# 加载模型
model = torchvision.models.resnet18(pretrained=True)
# print(model)

# data = torch.rand(1,3,2,3)
# print(data)
# 1行3列->里面的元素是2行3列
# [[
#     [
#         [0.4164, 0.6646, 0.3048],
#         [0.2324, 0.2872, 0.3802]
#     ],

#     [
#         [0.2340, 0.7861, 0.2441],
#         [0.2722, 0.7214, 0.0055]
#     ],

#     [
#         [0.4195, 0.6159, 0.6107],
#         [0.8148, 0.8508, 0.0753]
#     ]]]

data = torch.rand(1,3,64,64)
# print(data)

labels = torch.rand(1,1000)

# print(labels)

# 正向传播
prediction  = model(data)
# print(prediction)

# 计算模型的预测与标签的误差
loss = (prediction - labels ).sum()
# print(loss)       # tensor(-500.7379, grad_fn=<SumBackward0>)

# 通过网络反向传播
back = loss.backward()
print( back )   # None
print( loss )     # tensor(-500.7379, grad_fn=<SumBackward0>)         


# 加载优化器
# 1e-2 => 0.01
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# 数据的模型参数
# print( model.parameters() ) # <generator object Module.parameters at 0x000001F093C5AF90>
# for i in model.parameters():
#     print(i)


ret = optim.step()
print(ret)  # None

print(model)

'''
训练步骤：
1. 加载经过预训练数据模型
2. 正向传播进行数据预测
3. 计算误差，反向传播
4. 加载优化器
5. 训练

'''


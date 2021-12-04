import torch
import numpy as np

data = [
    [1,2],
    [3,4]
]

# 1. 直接量生成张量
x_data = torch.tensor(data)
print(f'data=>{data}\r\nx_data=>{x_data}')

'''
data=>[[1, 2], [3, 4]]
x_data=>tensor([[1, 2],
        [3, 4]])
'''
print('---------2----------')
# 2. Numpy的数组转换
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f'np_array=>{np_array}\r\nx_np=>{x_np}')
'''
np_array=>[[1 2]
 [3 4]]
x_np=>tensor([[1, 2],
        [3, 4]], dtype=torch.int32)
'''
# 张量的属性
print(x_np.shape,x_np.dtype, x_np.device)
print('----------3--------------')

# 3. 通过指定维度来生成张量
# 3行2列
shape = (3,2)
ones_tensor = torch.ones(shape)
print(ones_tensor)




# 当前环境GPU是否可用，然后将tensor导入GPU内运行
print(torch.cuda.is_available())


print('-'*10, '张量运算', '-'*10)


# 张量运算
tensor = torch.ones(4,4)
print(tensor)

# 把每一行的第一列变成99
tensor[:,1] = 99

# 把第2行的每一列变成22
tensor[2,:] = 22
print(tensor)


# 张量乘法(跟np一样是矩阵上对应元素相乘)
data1 = [
    [1,2],
    [3,4]
]
data2 = [
    [5,6],
    [7,8]
]

tensor1 = torch.tensor(data1)
tensor2 = torch.tensor(data2)

print(tensor1 * tensor2)
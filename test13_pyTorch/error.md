



# 分类训练遇到的`bug`
1. 通道问题

```shell
Traceback (most recent call last):
  File "D:\Github\openCV\test13_pyTorch\LeNet.py", line 121, in <module>
    outputs = net(inputs)
  File "D:\Programs\python3.9.8\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "D:\Github\openCV\test13_pyTorch\LeNet.py", line 79, in forward
    x = self.relu(self.conv1(x))
  File "D:\Programs\python3.9.8\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "D:\Programs\python3.9.8\lib\site-packages\torch\nn\modules\conv.py", line 446, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "D:\Programs\python3.9.8\lib\site-packages\torch\nn\modules\conv.py", line 442, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Given groups=1, weight of size [6, 1, 5, 5], expected input[1, 3, 32, 32] to have 1 channels, but got 3 channels instead
```

* 解决方案

> 标准化的时候，更改通道

```py
# 1. 加载并标准化
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Grayscale(num_output_channels=1)
     ])

```


2. `input size`

```shell
Traceback (most recent call last):
  File "D:\Github\openCV\test13_pyTorch\LeNet.py", line 134, in <module>
    outputs = net(inputs)
  File "D:\Programs\python3.9.8\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "D:\Github\openCV\test13_pyTorch\LeNet.py", line 98, in forward
    x = x.view(-1, 256)
RuntimeError: shape '[-1, 256]' is invalid for input of size 1600
```


* 解决方案
``shell
```


3. requests下载分类集时，`raise ConnectionError(err, request=request)
requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))`

* 解决方法：
```shell
# 设置代理

```
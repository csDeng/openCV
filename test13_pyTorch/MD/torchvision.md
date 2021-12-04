模块 `torchvision` 库包含了计算机视觉中一些常用的数据集, 模型架构以及图像变换方法.

Package Reference

- torchvision.datasets
  - [MNIST](https://pytorch.apachecn.org/#/datasets.html?id=mnist)
  - [Fashion-MNIST](https://pytorch.apachecn.org/#/datasets.html?id=fashion-mnist)
  - [COCO](https://pytorch.apachecn.org/#/datasets.html?id=coco)
  - [LSUN](https://pytorch.apachecn.org/#/datasets.html?id=lsun)
  - [ImageFolder](https://pytorch.apachecn.org/#/datasets.html?id=imagefolder)
  - [Imagenet-12](https://pytorch.apachecn.org/#/datasets.html?id=imagenet-12)
  - [CIFAR](https://pytorch.apachecn.org/#/datasets.html?id=cifar)
  - [STL10](https://pytorch.apachecn.org/#/datasets.html?id=stl10)
  - [SVHN](https://pytorch.apachecn.org/#/datasets.html?id=svhn)
  - [PhotoTour](https://pytorch.apachecn.org/#/datasets.html?id=phototour)
- torchvision.models
  - [Alexnet](https://pytorch.apachecn.org/#/models.html?id=id1)
  - [VGG](https://pytorch.apachecn.org/#/models.html?id=id2)
  - [ResNet](https://pytorch.apachecn.org/#/models.html?id=id3)
  - [SqueezeNet](https://pytorch.apachecn.org/#/models.html?id=id4)
  - [DenseNet](https://pytorch.apachecn.org/#/models.html?id=id5)
  - [Inception v3](https://pytorch.apachecn.org/#/models.html?id=inception-v3)
- torchvision.transforms
  - [PIL Image 上的变换](https://pytorch.apachecn.org/#/transforms.html?id=pil-image)
  - [torch.*Tensor 上的变换](https://pytorch.apachecn.org/#/transforms.html?id=torch-tensor)
  - [转换类型的变换](https://pytorch.apachecn.org/#/transforms.html?id=id1)
  - [通用的变换](https://pytorch.apachecn.org/#/transforms.html?id=id2)
- [torchvision.utils](https://pytorch.apachecn.org/#/utils.html)



-----



```py
torchvision.get_image_backend()复制ErrorOK!
```

获取用于加载图像的包的名称

```py
torchvision.set_image_backend(backend)复制ErrorOK!
```

指定用于加载图像的包.

参数：`backend (string)` – 图像处理后端的名称. {'PIL', 'accimage'} 之一. `accimage` 使用 Intel IPP library(高性能图像加载和增强程序模拟的程序）.通常比PIL库要快, 但是不支持许多操作.


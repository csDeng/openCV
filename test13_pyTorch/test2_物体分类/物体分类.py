import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# If running on Windows and you get a BrokenPipeError, try setting
# the num_worker of torch.utils.data.DataLoader() to 0.

'''
class torchvision.transforms.Normalize(mean, std) 功能：
    对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc

class torchvision.transforms.ToTensor 功能：
    将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 
    注意事项：归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。

class torchvision.transforms.Grayscale(num_output_channels=1) 功能：
    将图片转换为灰度图 
    参数： num_output_channels- (int) ，当为1时，正常的灰度图，当为3时， 3 channel with r == g == b
'''
# 1. 加载并标准化
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Grayscale(num_output_channels=1)
     ])


"""`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

Args:
    root (string): Root directory of dataset where directory
        ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
    train (bool, optional): If True, creates dataset from training set, otherwise
        creates from test set.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

"""
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# tool functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 2. 定义卷积神经网络
import torch.nn as nn
import torch.nn.functional as F
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        # nn.Conv2d将采用nSamples x nChannels x Height x Width的 4D 张量。
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        """
        Applies a linear transformation to the incoming data: :math:`y = xA^T + b
        input-feature
        output-feature
        """

        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)

        self.classifier = nn.Linear(84, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        """Applies the rectified linear unit function element-wise:
            :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
            Args:
                inplace: can optionally do the operation in-place. Default: ``False``
            Shape:
                - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
                - Output: :math:`(*)`, same shape as the input.
            .. image:: ../scripts/activation_images/ReLU.png

            Examples::
                >>> m = nn.ReLU()
                >>> input = torch.randn(2)
                >>> output = m(input)
        """
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        # 通过view函数来调整x的大小
        x = x.view(-1, 256)
        # x = x.view(-1, 1600) # 为解决bug做的调整
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.classifier(x)
        return x


net = Lenet()

# 检查网络模型参数(自己加的)
params = list(net.parameters())
print(len(params))
print(params[0].size())

print('-'*30)



# 3. 定义损失函数和优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 4. 训练网络
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # 看看输入的tensor是什么
        # print('input=>',inputs)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 4.1 保存训练过的模型
PATH = './cifar_net.pth'
torch.save(Lenet.state_dict(), PATH)

# 5. 根据测试数据测试网络

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))



# 6. 重新加载保存的模型
net = Lenet()
net.load_state_dict(torch.load(PATH))

# 7. 看看神经网络对以上示例的看法
outputs = net(images)


# 8. 让我们获取最高能量的指数
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))



# 9. 让我们看一下网络在整个数据集上的表现。
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# 10. 看看哪些类，正确率高
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



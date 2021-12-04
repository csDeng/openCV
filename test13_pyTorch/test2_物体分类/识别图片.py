from PIL import Image
import torch
import torchvision
import os
from torchvision import transforms
from matplotlib import pyplot as plt 
import requests
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# https://pytorch.org/hub/pytorch_vision_resnet/

def down_class():
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36 Edg/96.0.1054.41',
        "Content-type" : "application/json",
        'Connection': 'close', # 关闭持久连接
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
    }

    # 因为是github上面的资源，不开代理会连接超时
    proxies = {
        "http":"http://127.0.0.1:7890",
        "https":"http://127.0.0.1:7890"
    }
    
    # 下载分类数据集标签
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'

    try:
        r = requests.get(url, headers=headers, proxies=proxies)
        if r.status_code == 200:
            # print(r.content.decode('utf-8'))
            with open('./classes.txt',mode='a+', encoding='utf-8') as f:
                f.write(r.content.decode('utf-8'))
            
            print('*'*50, 'download successfully')
            return True
        raise Exception('get classes error is raised')
    except Exception as e:
        print('download classes error=>\r\n', e)

def existed_or_not():
    return os.path.exists('./classes.txt')
def get_classes():
    global classes
    if not existed_or_not():
        down_class()
    with open('./classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    # print(classes)

    
# 识别
'''
url=> 预测的图片的本地地址
'''
def check(url):

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
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])


    img = Image.open(url)
    plt.imshow(img)
    plt.show()
    img = trans(img)

    img = img.unsqueeze(0)

    #  pretrained (bool): If True, returns a model pre-trained on ImageNet
    # progress (bool): If True, displays a progress bar of the download to stderr
    model = torchvision.models.resnet18(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    
    # 将网络设置为eval模式 => 如果没有设置模式，归一化出来的预测结果不对
    model.eval()
    result = model(img)
    '''
    使用max()函数对softmax函数的输出值进行操作，求出预测值索引，然后与标签进行比对，计算准确率
    input是softmax函数输出的一个tensor
    dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值

    => 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
    '''
    # res = torch.max(result, 1)
    # print('result=>',result, label)

    # print('res=>',res)
    # 识别分类的时候，拿到索引即可

    _, index = torch.max(result, 1)
    print(f'index=>{index}')

    # 输出有未归一化的分数。为了得到概率，你可以对它进行软最大值测试。
    percentage = torch.nn.functional.softmax(result, dim=1)[0] * 100

    # 输出分类，以及概率
    print(classes[index[0]], percentage[index[0]].item())


def main():
    get_classes()
    imgs = {
        'human_test': 'human_test.jpg', 
        'human_test2':'human_test2.jpg', 
        'car':'car.jpg',
        'dog2':'dog.jpg',
        'plane':'plane.jpg'
    }
    for i in enumerate(imgs):
        url = imgs[ f'{i[1]}' ]
        url = './imgs/'+url
        # print( url )
        check( url )


if __name__ == '__main__':
    main()
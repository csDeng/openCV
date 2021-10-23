# 实验目的与要求

（一）熟悉图像的几何变换；

（二）掌握图像的二值化;

（三）掌握二值化图像处理的数学形态学滤波.



# 实验内容

（一）使用OpenCV对图像进行缩放、旋转、相似变换、仿射变换

缩放

![img](https://i.loli.net/2021/10/23/DiAVpekWxzJbG2u.jpg) 

翻转：

![img](https://i.loli.net/2021/10/23/mjQeJPHIru1Z5af.jpg) 

相似变换、仿射变换

![img](https://i.loli.net/2021/10/23/4nqYBCi75EpdL8W.jpg) 

源码：
```python

import cv2

import numpy as np

 

def run():

  try:

​    origin = cv2.imread('./rgb.bmp', cv2.IMREAD_UNCHANGED)

​    \# 把原图缩小1/2

​    origin = cv2.resize(origin, None, fx=0.5, fy=0.5)

​    cv2.imshow('origin', origin)

​    \# 1. 缩放

​    \# 使用临近插值

​    \# big = cv2.resize(origin, None, fx=1.2, fy=1.5, interpolation = cv2.INTER_NEAREST)

​    \# small = cv2.resize(origin, None, fx=0.8, fy=0.8)

​    \# cv2.imshow('big', big)

​    \# cv2.imshow('small', small)

​    \# 2. 翻转

​    \# x = cv2.flip(origin, 0)

​    \# y = cv2.flip(origin, 1)

​    \# xy = cv2.flip(origin, -1)

​    \# cv2.imshow('x', x)

​    \# cv2.imshow('y', y)

​    \# cv2.imshow('xy', xy)

​    \# 3. 相似变换

​    \# 3.1平移

​    height, width = origin.shape[:2]

​    x = 50

​    y = 100

​    M = np.float32([ [1,0,x], [0,1,y]])

​    move = cv2.warpAffine(origin, M, (width, height))

​    \# 3.2 旋转 + 尺度

​    M = cv2.getRotationMatrix2D( (width/2, height/2), 45, 0.6)

​    dst = cv2.warpAffine(move, M, (width, height))

​    cv2.imshow('like', dst)

​    \# 4. 仿射变换

​    rows, cols, ch = origin.shape

​    p1 = np.float32([ [0,0], [cols-1,0], [0, rows-1]])

​    p2 = np.float32([ [0, rows*0.33], [cols*0.85, rows*0.25], [cols*0.15, rows*0.7]])

​    m = cv2.getAffineTransform(p1, p2)

​    dst = cv2.warpAffine(origin, m, (cols, rows))

​    cv2.imshow('wrapAffine', dst)

​    cv2.waitKey()

​    cv2.destroyAllWindows()

  except Exception as e:

​    print('error=>\r\n',e)

if __name__ == '__main__':

  run()

 
```
 

（二）使用OpenCV对图像进行二值化，对比阈值为128和大津法阈值效果

![img](https://i.loli.net/2021/10/23/fn65HNsGVtR3eTA.jpg)源码：
```python

import cv2

import numpy as np

 

\# 使用OpenCV对图像进行二值化，对比阈值为128和大津法阈值效果

class Solution:

  def __init__(self, path):

​    print('Solution Object is created!!!')

​    \# print(type(createImg) is np.ndarray)

​    \# 读取到的图像

​    self.img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

​    self.binary = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

​    

​    \# 原图太大了，缩小一点点

​    self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5)

​    self.binary = cv2.resize(self.binary, None, fx=0.5, fy=0.5)

​    \# print(width, height)

​    cv2.imshow('origin', self.img)

​    \# 随机生成一个4x5的矩阵序列

​    \# self.img = np.random.randint(0, 256, size=[4,5], dtype=np.uint8)

  def thresh_binary(self):

​    \# 图像二值化

​    try:

​      retVal,dest = cv2.threshold(self.img, 128, 255, cv2.THRESH_BINARY)

​      print('img=>\r\n', self.img)

​      print('对比阈值=',retVal)

​      print('二值化处理结果=>\r\n', dest)

​      cv2.imshow('binary', dest)

​    except Exception as e:

​      print('thresh_binary error=>\r\n', e)

  def Otsu(self):

​    try:

​      t, otsu = cv2.threshold(self.binary, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

​      print('Otsu计算得到的最优阈值=', t)

​      print('大津化阈值处理结果=>\r\n', otsu)

​      cv2.imshow('Otsu', otsu) 

​    except Exception as e:

​      print('Otsu error=\r\n', e)

  

  def __del__(self):

​    cv2.waitKey()

​    cv2.destroyAllWindows()

if __name__ == '__main__':

  o = Solution('./test.jpg')

  o.thresh_binary()

  o.Otsu()

  del o

 
```

（三）使用OpenCV对二值图进行腐蚀、膨胀、开运算、闭运算

腐蚀，膨胀

![img](https://i.loli.net/2021/10/23/Rz96BpUPiZg71kv.jpg) 

源码：
```python

import cv2

import numpy as np

 

try:

  \# 1. 腐蚀

  img = cv2.imread('./test2.png', cv2.IMREAD_GRAYSCALE)

  kernel = np.ones((2,2), np.uint8)

  cv2.imshow('origin', img)

  erosion = cv2.erode(img, kernel)

  cv2.imshow('erosion', erosion)

  \# 2. 膨胀

  dilation = cv2.dilate(img, kernel)

  cv2.imshow('dilation', dilation)

  cv2.waitKey()

  cv2.destroyAllWindows()

except Exception as e: 

  print('error=>\r\n', e)

 
```
 

开运算

![img](https://i.loli.net/2021/10/23/DHj2xUVGwPnue5J.jpg) 

源码
```python
import cv2

import numpy as np

 

try:

  \# 3. 开运算

  img = cv2.imread('./test4.png', cv2.IMREAD_GRAYSCALE)

  kernel = np.ones((5,5), np.uint8)

cv2.imshow('origin', img)

 

\# 先腐蚀再膨胀

  erosion = cv2.erode(img, kernel)

  cv2.imshow('erosion', erosion)

  dilation = cv2.dilate(erosion, kernel)

cv2.imshow('dilation', dilation)

 

\# 直接调用函数

  morphologyEx = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

  cv2.imshow('morphologyEx', morphologyEx)

 

  cv2.waitKey()

  cv2.destroyAllWindows()

except Exception as e: 

  print('error=>\r\n', e)

 ```

闭运算

![img](https://i.loli.net/2021/10/23/2ZT7RwyrcMSm4WI.jpg) 

源码

```python

import cv2

import numpy as np

 

try:

​    \# 4. 闭运算

  img = cv2.imread('./test5.png', cv2.IMREAD_UNCHANGED)

  cv2.imshow('origin', img)

  kernel = np.ones((10,10), np.uint8)

  r = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

  cv2.imshow('result', r)

  cv2.waitKey()

  cv2.destroyAllWindows()

except Exception as e: 

  print('error=>\r\n', e)
  
  ```
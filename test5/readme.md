* 1. 使用OpenCV对图像计算梯度，分别使用Sobel和Laplacian算子
* 2. 使用OpenCV对图像Canny边缘检测，显示并保存

* 3. 使用OpenCV对house.tif进行霍夫直线检测，对硬币图片进行霍夫圆形检测





---



（一）使用OpenCV对图像计算梯度，分别使用Sobel和Laplacian算子

Sobel算子：

![img](https://i.loli.net/2021/10/23/nYQTlS5vLy6oRh4.jpg) 

源代码：

```python

import cv2

 

try:

  img = cv2.imread('./test.png', cv2.IMREAD_GRAYSCALE)

  \# print(img)

  img = cv2.resize(img, None, fx=0.5, fy=0.5 )

  cv2.imshow('origin', img)

  x = cv2.Sobel(img, cv2.CV_64F, 1,0 )

  y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

  xy = cv2.addWeighted(x, 0.5, y, 0.5, 0 )

  cv2.imshow('x', x)

  cv2.imshow('y', y)

  cv2.imshow('xy', xy)

  cv2.waitKey()

  cv2.destroyAllWindows()

except Exception as e:

  print('error=>', e)

```

Laplacian算子：

![img](https://i.loli.net/2021/10/23/31tXzvIogeTqUhA.jpg) 

源代码：

```python


import cv2

 

try:

  img = cv2.imread('./test.png', cv2.IMREAD_GRAYSCALE)

  \# print(img)

  img = cv2.resize(img, None, fx=0.5, fy=0.5 )

  cv2.imshow('origin', img)

  Laplacian = cv2.Laplacian(img, cv2.CV_64F)

  Laplacian = cv2.convertScaleAbs(Laplacian)

  cv2.imshow('Laplacian', Laplacian)

  cv2.waitKey()

  cv2.destroyAllWindows()

except Exception as e:

  print('error=>', e)

```


（二）使用OpenCV对图像Canny边缘检测，显示并保存

二值图的Canny处理

![img](https://i.loli.net/2021/10/23/wv5O9tUBZcpQTnx.jpg) 

彩色图像的Canny处理

![img](https://i.loli.net/2021/10/23/ZeStKRA481alzhB.jpg) 

保存结果：

![img](https://i.loli.net/2021/10/23/wzLJejv9OrqBoE5.jpg) 

源代码：

```python
import cv2

 

try:

  \# img = cv2.imread('./rgb.bmp', cv2.IMREAD_GRAYSCALE)

  img = cv2.imread('./rgb.bmp', cv2.IMREAD_UNCHANGED)

  \# print(img)

  img = cv2.resize(img, None, fx=0.5, fy=0.5 )

  cv2.imshow('origin', img)

  r1 = cv2.Canny(img, 128, 200)

  r2 = cv2.Canny(img, 32, 128)

  cv2.imshow('result1', r1)

  cv2.imshow('result2', r2)

  cv2.imwrite('result1.png', r1)

  cv2.imwrite('result2.png', r2)

  cv2.waitKey()

  cv2.destroyAllWindows()

  

except Exception as e:

  print('error=>', e)

```

（三）使用OpenCV对house.tif进行霍夫直线检测，对硬币图片进行霍夫圆形检测

霍夫直线变化

![img](https://i.loli.net/2021/10/23/zUIREkQX9ZAqy13.jpg) 

源代码

```python

\# 霍夫直线检测

import numpy as np

import matplotlib.pyplot as plt

import cv2

 

try:

  img = cv2.imread('./house.tif')

  \# cv2.imshow('origin', img)

  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  orgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  edges = cv2.Canny(gray, 50, 150, apertureSize=3)

  \# threshold阈值， 越小线越多

  lines = cv2.HoughLines(edges, 1, np.pi/360, threshold=50)

  oShow = orgb.copy()

  for line in lines:

    rho, theta = line[0]

    a = np.cos(theta)

    b = np.sin(theta)

   x0 = a * rho

   y0 = b * rho

   x1 = int(x0 + 1000*(-b))

    y1 = int(y0 + 1000*a)

    x2 = int(x0 - 1000*(-b))

    y2 = int(y0 - 1000*a)

    cv2.line(orgb, (x1,y1), (x2,y2), (0,0,255), 1)

  \# 均等地划分画布 plt.subplot(nrows, ncols, index)

  plt.subplot(121)

  plt.imshow(oShow)

  cv2.waitKey()

  \# 关闭坐标轴

  plt.axis('off')

  plt.subplot(122)

  plt.imshow(orgb)

  plt.axis('off')

  

  plt.show()

  cv2.waitKey()

  cv2.destroyAllWindows()

except Exception as e:

  print('Error=> \r\n', e)

```

概率霍夫变换

![img](https://i.loli.net/2021/10/23/yTe3wCQxhcgdzGs.jpg) 



源代码：

```python

\# 概率霍夫曼直线处理

 

\# 霍夫直线检测

import numpy as np

import matplotlib.pyplot as plt

import cv2

try:

  img = cv2.imread('./house.tif')

  \# cv2.imshow('origin', img)

  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  orgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  edges = cv2.Canny(gray, 50, 150, apertureSize=3)

  \# minLineLength 接受最小的直线长度, maxLineGap 共线线段之间的最小距离

  lines = cv2.HoughLinesP(edges, 1, np.pi/180, 0, minLineLength=30, maxLineGap=8)

  oShow = orgb.copy()

  for line in lines:

    x1,y1, x2,y2 = line[0]

    \# 画直线

    cv2.line(orgb, (x1,y1), (x2,y2), (255,0,0), 2)

  \# 均等地划分画布 plt.subplot(nrows, ncols, index)

  plt.subplot(121)

  plt.imshow(oShow)

  cv2.waitKey()

  \# 关闭坐标轴

  plt.axis('off')

  plt.subplot(122)

  plt.imshow(orgb)

  plt.axis('off')

  

  plt.show()

  cv2.waitKey()

  cv2.destroyAllWindows()

except Exception as e:

  print('Error=> \r\n', e)

```


硬币的霍夫圆变换

![img](https://i.loli.net/2021/10/23/pSqQ62GYPXTK7Cf.jpg) 

```python
\# 霍夫圆变换

import numpy as np

import matplotlib.pyplot as plt

import cv2

try:

  img = cv2.imread('./China.jpg', 0)

  imgo = cv2.imread('./China.jpg', -1)

  o = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)

  oshow = o.copy()

  img = cv2.medianBlur(img, 5)

  circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,

  param1 = 50, param2 = 30, minRadius=270, maxRadius=280)

  circles = np.uint16(np.around(circles))

  for i in circles[0,:]:

    cv2.circle(o, (i[0], i[1]) , i[2], (255,0,0),12) 

​    cv2.circle(o, (i[0], i[1]) , 2, (255,0,0),12) 

  \# 均等地划分画布 plt.subplot(nrows, ncols, index)

  plt.subplot(121)

  plt.imshow(oshow)

  cv2.waitKey()

  \# 关闭坐标轴

  plt.axis('off')

  plt.subplot(122)

  plt.imshow(o)

  plt.axis('off')

  plt.show()

except Exception as e:

  print('Error=> \r\n', e)

```
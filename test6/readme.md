（一）使用OpenCV对图像进行Harris，SIFT特征点提取，并标注特征点

```python
\# 使用OpenCV对图像进行Harris，SIFT特征点提取，并标注特征点

 

import cv2

import numpy as np

try:

  \# print(cv2.__version__)

  \# filename = "test.tiff"

  filename = "qipan.jpg"

  img = cv2.imread("./pics/"+filename)

  \# print(img)

  \# cv2.imshow("origin", img)

  o = img.copy()  # 深拷贝原图

  o1 = img.copy() # 深拷贝做sift

 

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  gray1 = gray.copy()   # 深拷贝给sift使用

 

  \# harries

  gray = np.float32(gray)

  harries = cv2.cornerHarris(gray, 2, 3, 0.04)

  \# img[harries>0.01*dst.max()]=[0,0,255] 

  \# 系数越小，识别到的角点越多

  \# 描点

  img[harries>0.01*harries.max()]=[255,0,0]

 

  \# SIFT

  sift = cv2.xfeatures2d.SIFT_create()

  kp = sift.detect(gray1, None)

  cv2.drawKeypoints(o1,kp,o1)

 

  \# 把两张图片一起显示

  imghStack = np.hstack((o, img, o1))

  cv2.imshow("o-harries-sift", imghStack)

  cv2.waitKey()

  cv2.destroyAllWindows()

except Exception as e:

  print("error=>",e)

 
```



![img](https://i.loli.net/2021/10/30/woyhpMNi8u1rvFD.jpg) 

（二）使用OpenCV生成特征的SIFT描述子，对两幅有重叠的图片进行描述子匹配

源码：

```python
\# （二）使用OpenCV生成特征的SIFT描述子，对两幅有重叠的图片进行描述子匹配

import cv2 as cv

import numpy as np

try:

 

  box = cv.imread("./pics/hanxin1.jpg")

  box_in_sence = cv.imread("./pics/hanxin.png")

  cv.imshow("box", box)

  cv.imshow("box_in_sence", box_in_sence)

  \# 创建SIFT特征检测器

  sift = cv.xfeatures2d.SIFT_create()

  kp1, des1 = sift.detectAndCompute(box,None)

  kp2, des2 = sift.detectAndCompute(box_in_sence,None)

  \# 暴力匹配

  bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)

  matches = bf.match(des1,des2)

  \# 绘制匹配

  matches = sorted(matches, key = lambda x:x.distance)

  '''

  img1 – 源图像1

  keypoints1 – 源图像1的特征点.

  img2 – 源图像2.

  keypoints2 – 源图像2的特征点

  matches1to2 – 源图像1的特征点匹配源图像2的特征点[matches[i]] .

  outImg – 输出图像具体由flags决定.

  matchColor – 匹配的颜色（特征点和连线),若matchColor==Scalar::all(-1)，颜色随机.

  singlePointColor – 单个点的颜色，即未配对的特征点，若matchColor==Scalar::all(-1)，颜色随机.

  matchesMask – Mask决定哪些点将被画出，若为空，则画出所有匹配点.

  flags – Fdefined by DrawMatchesFlags.

  '''

  result = cv.drawMatches(box, kp1, box_in_sence, kp2, matches1to2=matches[:30], outImg=None)

  \# 显示30个特征点

  cv.imshow("smallMatchBig", result)

  cv.waitKey()

  cv.destroyAllWindows()

except Exception as e:

  print("error=>",e)
```



结果

原图

![img](https://i.loli.net/2021/10/30/KGZ6kL5MfR13OHl.jpg) 

 

匹配的图

![img](https://i.loli.net/2021/10/30/9wEPCuDaOYMJksv.jpg) 

（三）使用OpenCV对两幅有重叠的图片匹配后进行拼接，生成全景图（选做）

源码

```python
"""

使用OpenCV对两幅有重叠的图片匹配后进行拼接，生成全景图

"""

 

import cv2 

import numpy as np

try:

  img1 = cv2.imread('./pics/1.png')

  img2 = cv2.imread('./pics/2.png')

  stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA) 

  (_result, pano) = stitcher.stitch((img1, img2))

  cv2.imshow("img1", img1)

  cv2.imshow("img2", img2)

  cv2.imshow('pano',pano)

  cv2.waitKey(0)

  cv2.destroyAllWindows()

except Exception as e:

  print("error====>\r\n", e, "\r\n ============\r\n")
```



拼接前

![img](https://i.loli.net/2021/10/30/BYIHvbuhDNjy9w1.jpg) 

拼接后

![img](https://i.loli.net/2021/10/30/BuwmjXOnfgqpEDz.jpg) 

 
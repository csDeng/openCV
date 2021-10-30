# （二）使用OpenCV生成特征的SIFT描述子，对两幅有重叠的图片进行描述子匹配
import cv2 as cv
import numpy as np
try:

    box = cv.imread("./pics/hanxin1.jpg")
    box_in_sence = cv.imread("./pics/hanxin.png")
    cv.imshow("box", box)
    cv.imshow("box_in_sence", box_in_sence)

    # 创建SIFT特征检测器
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(box,None)
    kp2, des2 = sift.detectAndCompute(box_in_sence,None)

    # 暴力匹配
    bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)

    matches = bf.match(des1,des2)

    # 绘制匹配
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
    # 显示30个特征点

    cv.imshow("smallMatchBig", result)

    cv.waitKey()
    cv.destroyAllWindows()


except Exception as e:
    print("error=>",e)
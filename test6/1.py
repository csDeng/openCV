# 使用OpenCV对图像进行Harris，SIFT特征点提取，并标注特征点

import cv2
import numpy as np
try:
    # print(cv2.__version__)
    # filename = "test.tiff"
    filename = "qipan.jpg"
    img = cv2.imread("./pics/"+filename)
    # print(img)
    # cv2.imshow("origin", img)
    o = img.copy()  # 深拷贝原图
    o1 = img.copy() # 深拷贝做sift

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = gray.copy()     # 深拷贝给sift使用

    # harries
    gray = np.float32(gray)
    harries = cv2.cornerHarris(gray, 2, 3, 0.04)
    # img[harries>0.01*dst.max()]=[0,0,255] 
    # 系数越小，识别到的角点越多
    # 描点
    img[harries>0.01*harries.max()]=[255,0,0]

    # SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray1, None)
    cv2.drawKeypoints(o1,kp,o1)


    # 把两张图片一起显示
    imghStack = np.hstack((o, img, o1))
    cv2.imshow("o-harries-sift", imghStack)

    cv2.waitKey()
    cv2.destroyAllWindows()


except Exception as e:
    print("error=>",e)
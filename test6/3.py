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
import cv2
import numpy as np

try:
    # # 1. 腐蚀
    # img = cv2.imread('./test2.png', cv2.IMREAD_GRAYSCALE)
    # kernel = np.ones((2,2), np.uint8)
    # cv2.imshow('origin', img)
    # erosion = cv2.erode(img, kernel)
    # cv2.imshow('erosion', erosion)

    # # 2. 膨胀
    # dilation = cv2.dilate(img, kernel)
    # cv2.imshow('dilation', dilation)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # 3. 开运算
    # img = cv2.imread('./test4.png', cv2.IMREAD_GRAYSCALE)
    # kernel = np.ones((5,5), np.uint8)
    # cv2.imshow('origin', img)
    # erosion = cv2.erode(img, kernel)
    # cv2.imshow('erosion', erosion)
    # dilation = cv2.dilate(erosion, kernel)
    # cv2.imshow('dilation', dilation)

    # morphologyEx = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('morphologyEx', morphologyEx)


    # 4. 闭运算
    img = cv2.imread('./test5.png', cv2.IMREAD_UNCHANGED)
    cv2.imshow('origin', img)
    kernel = np.ones((10,10), np.uint8)
    r = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('result', r)

    cv2.waitKey()
    cv2.destroyAllWindows()

except Exception as e: 
    print('error=>\r\n', e)

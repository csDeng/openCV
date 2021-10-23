import cv2


try:
    lena = cv2.imread('./rgb.bmp', cv2.IMREAD_UNCHANGED)
    # 均值滤波
    r1 = cv2.blur(lena, (5,5))

    # 高斯滤波
    r2 = cv2.GaussianBlur(lena, (5,5), 0,0)
    # 中值滤波
    r3 = cv2.medianBlur(lena, 3)

    cv2.imshow('origin', lena)
    cv2.imshow(u'blur', r1)
    cv2.imshow(u'gaussianBlur', r2)
    cv2.imshow(u'medianBlur', r3)

    cv2.waitKey()
    cv2.destroyAllWindows()
except Exception as e:
    print('error\r\n', e)
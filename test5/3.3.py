# 霍夫圆变换

# 概率霍夫曼直线处理


# 霍夫直线检测
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
        cv2.circle(o, (i[0], i[1]) , 2, (255,0,0),12) 
    # 均等地划分画布 plt.subplot(nrows, ncols, index)
    plt.subplot(121)
    plt.imshow(oshow)
    cv2.waitKey()

    # 关闭坐标轴
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(o)
    plt.axis('off')
    plt.show()
except Exception as e:
    print('Error=> \r\n', e)
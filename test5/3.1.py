# 霍夫直线检测
import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    img = cv2.imread('./house.tif')

    # cv2.imshow('origin', img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    orgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # threshold阈值， 越小线越多
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
    # 均等地划分画布 plt.subplot(nrows, ncols, index)
    plt.subplot(121)
    plt.imshow(oShow)
    cv2.waitKey()

    # 关闭坐标轴
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(orgb)
    plt.axis('off')
    
    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
except Exception as e:
    print('Error=> \r\n', e)
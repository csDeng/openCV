# 概率霍夫曼直线处理


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

    # minLineLength 接受最小的直线长度, maxLineGap 共线线段之间的最小距离
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 0, minLineLength=30, maxLineGap=8)

    oShow = orgb.copy()
    for line in lines:
        x1,y1, x2,y2 = line[0]

        # 画直线
        cv2.line(orgb, (x1,y1), (x2,y2), (255,0,0), 2)
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
import cv2
import matplotlib.pyplot as plt
try:
    lena = cv2.imread('rgb.bmp')
    gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    cv2.imshow('bgr', lena)
    cv2.imshow('gray', gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0,255])
    print('hist=\r\n', hist)
    # 开始画直方图
    #解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
    plt.figure()
    plt.title('灰度图像的直方图')
    # 直方图的横坐标
    plt.xlabel('pixl')
    # 直方图的纵坐标
    plt.ylabel('number')
    plt.plot(hist)
    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
except Exception as e:
    print('error\r\n',e)
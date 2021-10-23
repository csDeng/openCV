import cv2

try:
    # 读取图片
    lena = cv2.imread('./1.jpg', -1)
    # print(lena)

    # 展示读取到的图片
    cv2.namedWindow('lesson')
    cv2.imshow('l esson', lena)

    # 等待
    cv2.waitKey(0)

    # 销毁所有窗口
    cv2.destroyAllWindows()
except Exception as e:
    print(e)
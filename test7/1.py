import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        img = cv2.imread("./imgs/coins.jpg")
        # cv2.imshow("origin", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        isshow = img.copy()

        # 反二极化阈值处理
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # kernel算子
        kernel = np.ones((3,3), np.uint8)
        
        # 使用通用形态型函数进行开运算,去除噪声
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

        # 腐蚀
        sure_bg = cv2.dilate(opening, kernel, iterations = 3)

        # 距离变换
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # 确定前景区域
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)

        # 确定未知区域，减法运算
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标注边界
        ret, markers = cv2.connectedComponents(sure_fg)

        markers = markers + 1
        markers[unknown==255] = 0

        # 使用分水岭算法对预处理的图像进行处理
        markers = cv2.watershed(img, markers)

        img[markers == -1] = [255,0,0]

        # 在当前窗口添加子窗口
        plt.subplot(121)
        # 显示图像
        plt.imshow(isshow)

        # 关闭坐标抽显示
        plt.axis('off')

        plt.subplot(122)

        plt.imshow(img)

        plt.axis('off')

        # 一定要show，不然不显示
        plt.show()

    except Exception as e:
        print("main error=>", e)


if __name__ == "__main__":
    main()
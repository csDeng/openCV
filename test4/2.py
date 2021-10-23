import cv2
import numpy as np

# 使用OpenCV对图像进行二值化，对比阈值为128和大津法阈值效果
class Solution:
    def __init__(self, path):
        print('Solution Object is created!!!')
        # print(type(createImg) is np.ndarray)
        # 读取到的图像

        self.img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.binary = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # 原图太小了，放大两倍
        self.img = cv2.resize(self.img, None, fx=0.5, fy=0.5)
        self.binary = cv2.resize(self.binary, None, fx=0.5, fy=0.5)
        # print(width, height)
        cv2.imshow('origin', self.img)
        # 随机生成一个4x5的矩阵序列
        # self.img = np.random.randint(0, 256, size=[4,5], dtype=np.uint8)

    def thresh_binary(self):
        # 图像二值化
        try:
            retVal,dest = cv2.threshold(self.img, 128, 255, cv2.THRESH_BINARY)
            print('img=>\r\n', self.img)
            print('对比阈值=',retVal)
            print('二值化处理结果=>\r\n', dest)
            cv2.imshow('binary', dest)
        except Exception as e:
            print('thresh_binary error=>\r\n', e)

    def Otsu(self):
        try:
            t, otsu = cv2.threshold(self.binary, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            print('Otsu计算得到的最优阈值=', t)
            print('大津化阈值处理结果=>\r\n', otsu)
            cv2.imshow('Otsu', otsu) 
        except Exception as e:
            print('Otsu error=\r\n', e)
    
    def __del__(self):
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    o = Solution('./test.jpg')
    o.thresh_binary()
    o.Otsu()
    del o

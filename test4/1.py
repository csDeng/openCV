import cv2
import numpy as np
# （一）使用OpenCV对图像进行缩放、旋转、相似变换、仿射变换
def run():
    try:
        origin = cv2.imread('./rgb.bmp', cv2.IMREAD_UNCHANGED)
        # 把原图缩小1/2
        origin = cv2.resize(origin, None, fx=0.5, fy=0.5)
        cv2.imshow('origin', origin)

        # 1. 缩放
        # 使用临近插值
        # big = cv2.resize(origin, None, fx=1.2, fy=1.5, interpolation = cv2.INTER_NEAREST)
        # small = cv2.resize(origin, None, fx=0.8, fy=0.8)
        # cv2.imshow('big', big)
        # cv2.imshow('small', small)

        # 2. 翻转
        # x = cv2.flip(origin, 0)
        # y = cv2.flip(origin, 1)
        # xy = cv2.flip(origin, -1)
        # cv2.imshow('x', x)
        # cv2.imshow('y', y)
        # cv2.imshow('xy', xy)


        # 3. 相似变换
        # 3.1平移
        height, width = origin.shape[:2]
        x = 50
        y = 100
        M = np.float32([ [1,0,x], [0,1,y]])
        move = cv2.warpAffine(origin, M, (width, height))

        # 3.2 旋转 + 尺度
        M = cv2.getRotationMatrix2D( (width/2, height/2), 45, 0.6)
        dst = cv2.warpAffine(move, M, (width, height))

        cv2.imshow('like', dst)

        # 4. 仿射变换
        rows, cols, ch = origin.shape

        p1 = np.float32([ [0,0], [cols-1,0], [0, rows-1]])
        p2 = np.float32([ [0, rows*0.33], [cols*0.85, rows*0.25], [cols*0.15, rows*0.7]])
        m = cv2.getAffineTransform(p1, p2)
        dst = cv2.warpAffine(origin, m, (cols, rows))
        cv2.imshow('wrapAffine', dst)



        cv2.waitKey()
        cv2.destroyAllWindows()
    except Exception as e:
        print('error=>\r\n',e)


if __name__ == '__main__':
    run()
import cv2
import numpy as np
def gamma_trans(img, r):
    return np.power(img,r)
# # 加载图片，把图像元素的数据类型转换成浮点数，处于255，把元素数值控制在0-1之间
img = cv2.imread('./rgb.bmp', cv2.IMREAD_UNCHANGED).astype(np.float32)/255
cv2.imshow('origin', img)
# 执行Gamma矫正
l = [0.5, 0.8, 1.5]
for i in l:
    cv2.imshow('gamma'+str(i), gamma_trans(img, i))

cv2.waitKey()
cv2.destroyAllWindows()
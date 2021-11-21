# 统计均值背景建模


import cv2 
import time
import numpy as np

# 视频文件输入初始化 
filename = "./snow.mp4"

cap = cv2.VideoCapture(filename) 
# ret, frame = cap.read()
# prevframe = frame    #第一帧
total = 0
zhen = 0

while( cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("the end !!!!")
        break

    frame= frame.astype(np.float32)
    total += frame
    zhen += 1
    print("当前帧数",zhen)

img = total / zhen
img = img.astype(np.uint8)
cv2.imshow("image", img)
cv2.waitKey()
cv2.destroyAllWindows()

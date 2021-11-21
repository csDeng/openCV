import cv2 
import time
# 视频文件输入初始化  
filename = "./dog.mp4" 
cap = cv2.VideoCapture(filename) 
ret, frame = cap.read()
prevframe = frame    #第一帧

while( cap.isOpened()):
    ret, nextframe = cap.read()
    if not ret:
        print("the end !!!!")
        break
    diff = cv2.absdiff(prevframe, nextframe)
    cv2.imshow("diff", diff)
    
    cv2.waitKey()
    prevframe = nextframe

time.sleep(2)
cv2.destroyAllWindows()

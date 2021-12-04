# 背景差分提取前景目标
import cv2
import numpy as np


# 背景建模
def build(filename):
    '''
    使用统计均值法
    '''
    cap = cv2.VideoCapture(filename) 
    # ret, frame = cap.read()
    # prevframe = frame    #第一帧
    total = 0
    zhen = 0

    while( cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("build is end !!!!")
            break

        frame= frame.astype(np.float32)
        total += frame
        zhen += 1
        # print("当前帧数",zhen)

    img = total / zhen
    img = img.astype(np.uint8)
    cap.release()
    return img

# 前景检测
def check(filename, backgroup):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret :
            print("check is end!!!")
            break
        diff = cv2.absdiff(backgroup, frame)
        cv2.imshow("diff", diff)
        makeTag(diff)
        cv2.waitKey()
     
    cap.release()
    cv2.destroyAllWindows()

# 轮廓信息标识
def makeTag(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 阈值处理
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 获取轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 画出tag
    for c in contours:
    #计算各轮廓的周长
        perimeter = cv2.arcLength(c,True)
        if perimeter > 188:
            #找到一个直矩形（不会旋转）
            x,y,w,h = cv2.boundingRect(c)
            #画出这个矩形
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    mask = np.zeros(img.shape, np.uint8)

    mask = cv2.drawContours(mask, contours, -1, (255,255,255),-1)

    # cv2.imshow("mask", mask)
    loc = cv2.bitwise_and(img,mask)
    cv2.imshow("Tag", img)


def main():
    filename = './snow.mp4'
    backgroup = build(filename)
    # cv2.imshow("backgroup", backgroup)
    # print(backgroup)
    # cv2.waitKey()
    check(filename, backgroup)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
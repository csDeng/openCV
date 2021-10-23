import numpy as np
import cv2

try:
    # 创建底图
    img = np.zeros((300,300,3) , dtype='uint8')
    img[:, :, :] = 255


    # 画直线
    cv2.line(img,(0,0),(300,300),(114, 141, 216))
    # 画蓝色矩形
    cv2.rectangle(img, (100,100), (200,200), (255,0,0))

    # 画红色圆形
    cv2.circle(img, (150,150), 50, (0,0,255))

    # 写文字
    font = cv2.FONT_ITALIC
    cv2.putText(img, "by~dcs", (100,298), font,1, (0,0,0))
    cv2.imshow("result", img)
    cv2.waitKey()
    if(cv2.imwrite('mypic.png', img)):
        print("保存成功")
    else:
        print("保存失败")

except Exception as e:
    print("error\r\n", e)
    cv2.destroyAllWindows()
    exit(-1)
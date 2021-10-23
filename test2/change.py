import numpy as np
import cv2

try:
    img = cv2.imread('./rgb.bmp', -1)

    size = len(img)

    st = int( (size/8)*3 )
    en = int( (size/8)*5 ) + 1 
    # print(st,en)
    for i in range(st, en):
        for j in range(st, en):
            img[i,j] = [0,0,0]
    print(img)
    cv2.imshow("image", img)
    cv2.waitKey()
    if( cv2.imwrite('result.png', img)):
        print("保存成功")
        cv2.destroyAllWindows()
    else:
        print("保存失败")
        exit(-1)

except Exception as e:
    print("error\r\n", e)
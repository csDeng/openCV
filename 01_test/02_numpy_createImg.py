import numpy as np

import cv2

try:
    blue = np.zeros((300,300,3), dtype='uint8')
    blue[:,:, 0] = 255
    # print("blue=\r\n", blue)


    # 画一条红线
    cv2.line(blue, (0,0), (300,300),(0,0,0))
    cv2.imshow("blue", blue)
    cv2.waitKey()
    cv2.destroyAllWindows()

except Exception as e:
    print("errorr\r\n", e)
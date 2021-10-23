import cv2
lena = cv2.imread('./rgb.bmp')
hsv = cv2.cvtColor(lena, cv2.COLOR_BGR2HSV)
yuv = cv2.cvtColor(lena, cv2.COLOR_BGR2YUV)
cv2.imshow('bgr', lena)
cv2.imshow('hsv', hsv)
cv2.imshow('yuv', yuv)
if cv2.imwrite('HSV.png', hsv) and cv2.imwrite('YUV.png', yuv):
    print('图像保存成功')
else:
    print('保存失败')
cv2.waitKey()
cv2.destroyAllWindows()
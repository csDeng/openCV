import cv2

a = cv2.imread('./rgb.bmp')
b = a
c = a+b
d = cv2.add(a,b)
cv2.imshow('lena', a)
cv2.imshow('+', c)
cv2.imshow('cv2.add', d)
cv2.waitKey()
cv2.destoyAllWindows()
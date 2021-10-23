import cv2

try:
    img = cv2.imread('./test.png', cv2.IMREAD_GRAYSCALE)
    # print(img)
    img = cv2.resize(img, None, fx=0.5, fy=0.5 )
    cv2.imshow('origin', img)
    x = cv2.Sobel(img, cv2.CV_64F, 1,0 )
    y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    xy = cv2.addWeighted(x, 0.5, y, 0.5, 0 )
    cv2.imshow('x', x)
    cv2.imshow('y', y)
    cv2.imshow('xy', xy)
    cv2.waitKey()
    cv2.destroyAllWindows()

except Exception as e:
    print('error=>', e)
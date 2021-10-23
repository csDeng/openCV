import cv2

try:
    img = cv2.imread('./test.png', cv2.IMREAD_GRAYSCALE)
    # print(img)
    img = cv2.resize(img, None, fx=0.5, fy=0.5 )
    cv2.imshow('origin', img)
    Laplacian = cv2.Laplacian(img, cv2.CV_64F)
    Laplacian = cv2.convertScaleAbs(Laplacian)
    cv2.imshow('Laplacian', Laplacian)
    cv2.waitKey()
    cv2.destroyAllWindows()

except Exception as e:
    print('error=>', e)
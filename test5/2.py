import cv2

try:
    # img = cv2.imread('./rgb.bmp', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('./rgb.bmp', cv2.IMREAD_UNCHANGED)
    # print(img)
    img = cv2.resize(img, None, fx=0.5, fy=0.5 )
    cv2.imshow('origin', img)
    r1 = cv2.Canny(img, 128, 200)
    r2 = cv2.Canny(img, 32, 128)
    cv2.imshow('result1', r1)
    cv2.imshow('result2', r2)
    cv2.imwrite('result1.png', r1)
    cv2.imwrite('result2.png', r2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
except Exception as e:
    print('error=>', e)
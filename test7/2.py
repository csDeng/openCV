import cv2
import numpy as np
import matplotlib.pyplot as plt
# 不使用掩模时：
def demo1():
    o = cv2.imread('./imgs/lena_color_512.tif', cv2.IMREAD_COLOR)
    orgb = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)

    mask = np.zeros(o.shape[:2], np.uint8)


    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 前景矩形框，注意宽度，不同宽度识别到的前景不一致
    rect = (50, 50, 400, 500)  

    cv2.grabCut(o, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2) | (mask==1), 0, 1).astype('uint8')
    ogc = o*mask2[:, :, np.newaxis]
    ogc = cv2.cvtColor(ogc, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(orgb)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(ogc)
    plt.show()

# 使用批注提取前景
def demo2():
    o = cv2.imread('./imgs/rgb.bmp')
    orgb = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)

    mask = np.zeros(o.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (50, 50, 400, 500)  

    cv2.grabCut(o, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    # 掩模 
    # 单通道灰度图
    mask2 = cv2.imread('./imgs/m6.jpg', 0)

    # 原格式
    mask2show = cv2.imread('./imgs/m6.jpg', -1)

    m2rgb = cv2.cvtColor(mask2show, cv2.COLOR_BGR2RGB)
    mask[ mask2 == 0 ] = 0
    # 确定背景

    mask[ mask2 == 255 ] = 1
    # 确定前景


    mask, bgd, fgd = cv2.grabCut(o, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    
    mask = np.where((mask==2) | (mask==1), 0, 1).astype('uint8')
    ogc = o*mask[:, :, np.newaxis]
    ogc = cv2.cvtColor(ogc, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(m2rgb)
    plt.axis('off')
    # plt.subplot(122)
    # plt.imshow(mask)
    plt.subplot(122)
    plt.imshow(ogc)
    plt.axis('off')
    plt.show()

def main():
    demo1()
    demo2()

if __name__ == "__main__":
    main()
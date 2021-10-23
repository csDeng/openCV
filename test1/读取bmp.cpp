#include<iostream>
#include<cstring>
using namespace std;

typedef struct _fileHeader{
    unsigned int   bfSize;        // 文件大小 以字节为单位(2-5字节)
    unsigned short bfReserved1;   // 保留，必须设置为0 (6-7字节)
    unsigned short bfReserved2;   // 保留，必须设置为0 (8-9字节)
    unsigned int   bfOffBits;     // 从文件头到像素数据的偏移  (10-13字节)
} BITMAPFILEHEADER;

typedef struct _imgHeader{
    unsigned int    biSize;          // 此结构体的大小 (14-17字节)
    long            biWidth;         // 图像的宽  (18-21字节)
    long            biHeight;        // 图像的高  (22-25字节)
    unsigned short  biPlanes;        // 表示bmp图片的平面属，显然显示器只有一个平面，所以恒等于1 (26-27字节)
    unsigned short  biBitCount;      // 一像素所占的位数，一般为24   (28-29字节)
    unsigned int    biCompression;   // 说明图象数据压缩的类型，0为不压缩。 (30-33字节)
    unsigned int    biSizeImage;     // 像素数据所占大小, 这个值应该等于上面文件头结构中bfSize-bfOffBits (34-37字节)
    long            biXPelsPerMeter; // 说明水平分辨率，用象素/米表示。一般为0 (38-41字节)
    long            biYPelsPerMeter; // 说明垂直分辨率，用象素/米表示。一般为0 (42-45字节)
    unsigned int    biClrUsed;       // 说明位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）。 (46-49字节)
    unsigned int    biClrImportant;  // 说明对图象显示有重要影响的颜色索引的数目，如果是0，表示都重要。(50-53字节)
} BITMAPINFOHEADER;

typedef struct _pixelInfo {
    unsigned char rgbBlue;   //该颜色的蓝色分量  (值范围为0-255)
    unsigned char rgbGreen;  //该颜色的绿色分量  (值范围为0-255)
    unsigned char rgbRed;    //该颜色的红色分量  (值范围为0-255)
    unsigned char rgbReserved;// 保留，必须为0
} PixelInfo;


BITMAPFILEHEADER fileHeader;
BITMAPINFOHEADER infoHeader;


void showBmpHead(BITMAPFILEHEADER pBmpHead)
{  //定义显示信息的函数，传入文件头结构体
    printf("BMP文件大小：%dkb\n", fileHeader.bfSize/1024);
    printf("保留字必须为0：%d\n",  fileHeader.bfReserved1);
    printf("保留字必须为0：%d\n",  fileHeader.bfReserved2);
    printf("实际位图数据的偏移字节数: %d\n",  fileHeader.bfOffBits);
}
void showBmpInfoHead(BITMAPINFOHEADER pBmpinfoHead)
{//定义显示信息的函数，传入的是信息头结构体
   printf("位图信息头:\n" );
   printf("信息头的大小:%d\n" ,infoHeader.biSize);
   printf("位图宽度:%d\n" ,infoHeader.biWidth);
   printf("位图高度:%d\n" ,infoHeader.biHeight);
   printf("图像的位面数(位面数是调色板的数量,默认为1个调色板):%d\n" ,infoHeader.biPlanes);
   printf("每个像素的位数:%d\n" ,infoHeader.biBitCount);
   printf("压缩方式:%d\n" ,infoHeader.biCompression);
   printf("图像的大小:%d\n" ,infoHeader.biSizeImage);
   printf("水平方向分辨率:%d\n" ,infoHeader.biXPelsPerMeter);
   printf("垂直方向分辨率:%d\n" ,infoHeader.biYPelsPerMeter);
   printf("使用的颜色数:%d\n" ,infoHeader.biClrUsed);
   printf("重要颜色数:%d\n" ,infoHeader.biClrImportant);
}



int main(){
    FILE * fp;
    fp = fopen("test1.bmp", "rb");
    if( fp==NULL ){
        cout << "图片打开失败" << endl;
        return -1;
    }

        
    //如果不先读取bifType，根据C语言结构体Sizeof运算规则——整体大于部分之和，从而导致读文件错位
    unsigned short  fileType;
    fread(&fileType,1,sizeof (unsigned short), fp);  
    if (fileType = 0x4d42)
    {
        printf("文件打开成功!" );  
        printf("\n文件标识符：%d\n", fileType);
        fread(&fileHeader, 1, sizeof(BITMAPFILEHEADER), fp);
        showBmpHead(fileHeader);
        cout << "=========" << endl;
        fread(&infoHeader, 1, sizeof(BITMAPINFOHEADER), fp);
        showBmpInfoHead(infoHeader);
        fclose(fp);
    }
    return 0; 

}

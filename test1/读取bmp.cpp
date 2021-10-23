#include<iostream>
#include<cstring>
using namespace std;

typedef struct _fileHeader{
    unsigned int   bfSize;        // �ļ���С ���ֽ�Ϊ��λ(2-5�ֽ�)
    unsigned short bfReserved1;   // ��������������Ϊ0 (6-7�ֽ�)
    unsigned short bfReserved2;   // ��������������Ϊ0 (8-9�ֽ�)
    unsigned int   bfOffBits;     // ���ļ�ͷ���������ݵ�ƫ��  (10-13�ֽ�)
} BITMAPFILEHEADER;

typedef struct _imgHeader{
    unsigned int    biSize;          // �˽ṹ��Ĵ�С (14-17�ֽ�)
    long            biWidth;         // ͼ��Ŀ�  (18-21�ֽ�)
    long            biHeight;        // ͼ��ĸ�  (22-25�ֽ�)
    unsigned short  biPlanes;        // ��ʾbmpͼƬ��ƽ��������Ȼ��ʾ��ֻ��һ��ƽ�棬���Ժ����1 (26-27�ֽ�)
    unsigned short  biBitCount;      // һ������ռ��λ����һ��Ϊ24   (28-29�ֽ�)
    unsigned int    biCompression;   // ˵��ͼ������ѹ�������ͣ�0Ϊ��ѹ���� (30-33�ֽ�)
    unsigned int    biSizeImage;     // ����������ռ��С, ���ֵӦ�õ��������ļ�ͷ�ṹ��bfSize-bfOffBits (34-37�ֽ�)
    long            biXPelsPerMeter; // ˵��ˮƽ�ֱ��ʣ�������/�ױ�ʾ��һ��Ϊ0 (38-41�ֽ�)
    long            biYPelsPerMeter; // ˵����ֱ�ֱ��ʣ�������/�ױ�ʾ��һ��Ϊ0 (42-45�ֽ�)
    unsigned int    biClrUsed;       // ˵��λͼʵ��ʹ�õĲ�ɫ���е���ɫ����������Ϊ0�Ļ�����˵��ʹ�����е�ɫ����� (46-49�ֽ�)
    unsigned int    biClrImportant;  // ˵����ͼ����ʾ����ҪӰ�����ɫ��������Ŀ�������0����ʾ����Ҫ��(50-53�ֽ�)
} BITMAPINFOHEADER;

typedef struct _pixelInfo {
    unsigned char rgbBlue;   //����ɫ����ɫ����  (ֵ��ΧΪ0-255)
    unsigned char rgbGreen;  //����ɫ����ɫ����  (ֵ��ΧΪ0-255)
    unsigned char rgbRed;    //����ɫ�ĺ�ɫ����  (ֵ��ΧΪ0-255)
    unsigned char rgbReserved;// ����������Ϊ0
} PixelInfo;


BITMAPFILEHEADER fileHeader;
BITMAPINFOHEADER infoHeader;


void showBmpHead(BITMAPFILEHEADER pBmpHead)
{  //������ʾ��Ϣ�ĺ����������ļ�ͷ�ṹ��
    printf("BMP�ļ���С��%dkb\n", fileHeader.bfSize/1024);
    printf("�����ֱ���Ϊ0��%d\n",  fileHeader.bfReserved1);
    printf("�����ֱ���Ϊ0��%d\n",  fileHeader.bfReserved2);
    printf("ʵ��λͼ���ݵ�ƫ���ֽ���: %d\n",  fileHeader.bfOffBits);
}
void showBmpInfoHead(BITMAPINFOHEADER pBmpinfoHead)
{//������ʾ��Ϣ�ĺ��������������Ϣͷ�ṹ��
   printf("λͼ��Ϣͷ:\n" );
   printf("��Ϣͷ�Ĵ�С:%d\n" ,infoHeader.biSize);
   printf("λͼ���:%d\n" ,infoHeader.biWidth);
   printf("λͼ�߶�:%d\n" ,infoHeader.biHeight);
   printf("ͼ���λ����(λ�����ǵ�ɫ�������,Ĭ��Ϊ1����ɫ��):%d\n" ,infoHeader.biPlanes);
   printf("ÿ�����ص�λ��:%d\n" ,infoHeader.biBitCount);
   printf("ѹ����ʽ:%d\n" ,infoHeader.biCompression);
   printf("ͼ��Ĵ�С:%d\n" ,infoHeader.biSizeImage);
   printf("ˮƽ����ֱ���:%d\n" ,infoHeader.biXPelsPerMeter);
   printf("��ֱ����ֱ���:%d\n" ,infoHeader.biYPelsPerMeter);
   printf("ʹ�õ���ɫ��:%d\n" ,infoHeader.biClrUsed);
   printf("��Ҫ��ɫ��:%d\n" ,infoHeader.biClrImportant);
}



int main(){
    FILE * fp;
    fp = fopen("test1.bmp", "rb");
    if( fp==NULL ){
        cout << "ͼƬ��ʧ��" << endl;
        return -1;
    }

        
    //������ȶ�ȡbifType������C���Խṹ��Sizeof������򡪡�������ڲ���֮�ͣ��Ӷ����¶��ļ���λ
    unsigned short  fileType;
    fread(&fileType,1,sizeof (unsigned short), fp);  
    if (fileType = 0x4d42)
    {
        printf("�ļ��򿪳ɹ�!" );  
        printf("\n�ļ���ʶ����%d\n", fileType);
        fread(&fileHeader, 1, sizeof(BITMAPFILEHEADER), fp);
        showBmpHead(fileHeader);
        cout << "=========" << endl;
        fread(&infoHeader, 1, sizeof(BITMAPINFOHEADER), fp);
        showBmpInfoHead(infoHeader);
        fclose(fp);
    }
    return 0; 

}

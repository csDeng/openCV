#include<iostream>
#include<cstring>
#include<cstdlib>
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
    const char * name = "rgb.bmp";
    fp = fopen(name, "rb");
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
        
        cout << "=============" <<endl;
        
        
        //ͼ�����ݵĲ���
		fseek(fp, fileHeader.bfOffBits, SEEK_SET);
		
		unsigned char *r, *g, *b;
		long w = infoHeader.biWidth;
		long h = infoHeader.biHeight;
		cout << w << h <<endl;
		r = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
		b = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
		g = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
		

		

		int i, j;
		unsigned char pixVal = '\0';
		for (i = 0; i < h; i++)
		{
			for (j = 0; j < w; j++)
			{	
				fread(&pixVal, sizeof(unsigned char), 1, fp);
				*(r + w * i + j) = pixVal;
				fread(&pixVal, sizeof(unsigned char), 1, fp);
				*(g + w * i + j) = pixVal;
				fread(&pixVal, sizeof(unsigned char), 1, fp);
				*(b + w * i + j) = pixVal;
			}
		}
		fclose(fp);
		//�洢ͼ��
		FILE* fpout;
		fpout = fopen("wb1.bmp", "wb");
		
		//	 д���ļ�ͷ����	
		fwrite(&fileType, sizeof(unsigned short), 1, fpout);		
		fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, fpout);
		fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, fpout);
		int flag = 0;
		for (j = 0; j < h; j++)
		{
			// ע��Ƚϵ�ʱ��Ҫ��С���� �÷�����ȫ���õ�0			
			if( j>= (0.375)*h && j<= (0.625)*h){
				flag = 1;
			}else{
				flag = 0;
			}
			for (i = 0; i < w; i++)
			{
				if( ( flag==1 ) && ( i >= (0.375)*w ) && ( i <= (0.625)*w ) ){
					
					pixVal = '0';
					fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
					fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
					fwrite(&pixVal, sizeof(unsigned char), 1, fpout);	
				}else{
					pixVal = r[w * j + i];
					fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
					pixVal = g[w * j + i];
					fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
					pixVal = b[w * j + i];
					fwrite(&pixVal, sizeof(unsigned char), 1, fpout);
				}

			}
		}
		fclose(fpout);
	}
	return 0;

}

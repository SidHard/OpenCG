//入口函数

#include "OpenCG.h"
#include "opencv.hpp"

using namespace CG;
using namespace cv;

#if (!defined USING_CMAKE) 
#pragma comment( lib, "opencv_core244d.lib" )
#pragma comment( lib, "opencv_highgui244d.lib" )
#pragma comment( lib, "opencv_imgproc244d.lib" )
#endif

Mat onMouseImg;
char dispChar[20];
void on_mouse(int event, int x,int y,int flags,void* param)  
{  
	if(event==CV_EVENT_MOUSEMOVE)//鼠标状态  
	{  
		Mat dispImg;
		cvtColor(onMouseImg, dispImg, CV_GRAY2RGB); 
		int gray = *(onMouseImg.data+y*(onMouseImg.cols)+x); 
		sprintf(dispChar,"x:%d, y:%d, gray:%d",x, y, gray);
		putText(dispImg,dispChar,Point(x+50,y), CV_FONT_HERSHEY_COMPLEX, 0.5 ,cvScalar(255,255,0));
		imshow("result",dispImg);  
	} 
}

void main()
{
	Mat m_Image = imread("D:\\1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	resize(m_Image, m_Image, Size(320, 240) );

	m_Image.convertTo(m_Image,CV_32FC1,1.0/255);

	imshow("source", m_Image);

	//////////////////////////////////
	//新建ImgIn，并从Mat中拷贝数据
	CG::Core::CGImage<float> *ImgIn = new CG::Core::CGImage<float>(m_Image.cols, m_Image.rows);
	//CG::Core::CGImage<float> *ImgDst = new CG::Core::CGImage<float>(m_Image.cols, m_Image.rows);
	CG::Core::CGImage<float> *ImgDst = new CG::Core::CGImage<float>((m_Image.cols / 4 - 1)*2*9, (m_Image.rows / 4 - 1)*2);
	CG::Core::CGImage<float> *ImgNorm = new CG::Core::CGImage<float>(m_Image.cols, m_Image.rows);
	CG::Core::CGImage<float> *ImgGrad = new CG::Core::CGImage<float>(m_Image.cols, m_Image.rows);

	float *imgData = ImgIn->GetData(false);

	memcpy(imgData, (float*)m_Image.data, m_Image.cols * m_Image.rows * sizeof(float));

	ImgIn->UpdateDeviceFromHost();
	//////////////////////////////////
	long s_t = getTickCount();
	//CG::Core::CGFilter(ImgDst, ImgIn, 0.1, 10, 1);
	//CG::Core::CGComputeGradient(ImgDst, ImgIn);
	//CG::Core::CGPyramid(ImgDst, ImgIn, 1.23);
	CG::Core::CGComputeGradNorm(ImgGrad, ImgNorm, ImgIn);
	CG::Hog::CGHogHistogram(ImgDst, ImgGrad, ImgNorm);
	long e_t = getTickCount();
	printf("%d", e_t - s_t);
	////////////////////////////////////
	ImgDst->UpdateHostFromDevice();

	Mat resultImg(ImgDst->hight, ImgDst->width, CV_32FC1, ImgDst->GetData(false));

	resultImg.convertTo(resultImg, CV_8UC1, 255);

	onMouseImg = resultImg;

	imshow("result", resultImg);

	cvSetMouseCallback("result",on_mouse);                         //鼠标回调，调试用

	cvWaitKey();

	ImgIn->Free();
	ImgDst->Free();
}
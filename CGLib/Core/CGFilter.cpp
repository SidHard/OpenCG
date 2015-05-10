//author 2015 Wang Xinbo

#include "CGFilter.h"
//#include "opencv.hpp"

//using namespace cv;
using namespace CG;
using namespace CG::Core;

float GaussianKernel[50];

float heuclideanLen(float a, float b, float d)
{
    float mod = (b - a) * (b - a);

    return exp(-mod / (2 * d * d));
}

float ucharToFloat(unsigned int c)
{
    float a;
    a = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    return a;
}

unsigned int floatToUchar(float a)
{
    unsigned char x = ((unsigned char)(fabs(a) * 255.0f)) & 0xff;

    return (x);
}

void Core::ConstructKernel_CPU(int radius, float delta)
{
    for (int i = 0; i < 2 * radius + 1; i++)
    {
        int x = i - radius;
        GaussianKernel[i] = exp(-(x * x) / (2 * delta * delta));
    }
}

void Core::CGFilter_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float e_d, int radius, int iterations)
{
	//准备数据
	float domainDist, dataDist, factor;
	float *pSrc = new float[ImgIn->dataSize];
	float *hImage = new float[ImgIn->dataSize];
	memcpy(pSrc, ImgIn->GetData(false), ImgIn->dataSize*sizeof(float));

	ConstructKernel_CPU(radius);

	for (int y = 0; y < ImgIn->hight; y++)
    {
		for (int x = 0; x < ImgIn->width; x++)
        {
            hImage[y * ImgIn->width + x] = pSrc[y * ImgIn->width + x];
        }
    }

	//进行滤波
	for (int y = 0; y < ImgIn->hight; y++)
    {
        for (int x = 0; x < ImgIn->width; x++)
        {
            float t = 0.0f;
            float sum = 0.0f;

            for (int i = -radius; i <= radius; i++)
            {
                int neighborY = y + i;

                //边界条件，避免溢出
                if (neighborY < 0)
                {
                    neighborY = 0;
                }
                else if (neighborY >= ImgIn->hight)
                {
                    neighborY = ImgIn->hight - 1;
                }

                for (int j = -radius; j <= radius; j++)
                {
                    domainDist = GaussianKernel[radius + i] * GaussianKernel[radius + j];

                    //clamp the neighbor pixel, prevent overflow
                    int neighborX = x + j;

                    if (neighborX < 0)
                    {
                        neighborX = 0;
                    }
                    else if (neighborX >= ImgIn->width)
                    {
                        neighborX = ImgIn->width - 1;
                    }

                    dataDist = heuclideanLen(hImage[neighborY * ImgIn->width + neighborX], hImage[y * ImgIn->width + x], e_d);
                    factor = domainDist * dataDist;
                    sum += factor;
                    t += factor * hImage[neighborY * ImgIn->width + neighborX];
                }
            }

            pSrc[y * ImgIn->width + x] = t/sum;
        }
    }

	memcpy(ImgDst->GetData(false), pSrc, ImgIn->width * ImgIn->hight * sizeof(float));

	delete[] pSrc;
	delete[] hImage;
}

void Core::CGFilter(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float e_d, int radius, int iterations)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGFilter_CUDA(ImgDst, ImgIn, e_d, radius, iterations);
#else
	CGFilter_CPU(ImgDst, ImgIn, e_d, radius, iterations);
#endif
}
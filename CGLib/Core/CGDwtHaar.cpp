//author 2015 Wang Xinbo

#include "CGDwtHaar.h"

using namespace CG;
using namespace CG::Core;

void Core::CGDwtHaar_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn, int haar_level)
{
	//准备数据
	float *pSrc = new float[ImgIn->dataSize];
	float *hImage = new float[ImgIn->dataSize];
	memcpy(pSrc, ImgIn->GetData(false), ImgIn->dataSize*sizeof(float));

	for(int i = 0; i < haar_level; i++)
	{
		memcpy(hImage, pSrc, ImgIn->dataSize*sizeof(float));
		//row
		for (int y = 0; y < ImgIn->hight; y++)
		{
			for (int x = 0; x < ImgIn->width; x++)
			{
				int globalPos = y * ImgIn->width + x;
				int midx = ImgIn->width/2;
				if(x < midx)
				{
					pSrc[globalPos] = (hImage[y * ImgIn->width + 2 * x + 1] + hImage[y * ImgIn->width + 2 * x]) * INV_SQRT_2;
				}
				else
				{
					int neighborX = 2 * (x - midx) + 1;
					if(neighborX >= ImgIn->width)
						neighborX = ImgIn->width - 1;
					pSrc[globalPos] = (hImage[y * ImgIn->width + neighborX] - hImage[y * ImgIn->width + 2 * (x - midx)]) * INV_SQRT_2;
				}
			}
		}

		memcpy(hImage, pSrc, ImgIn->dataSize*sizeof(float));
		//col
		for (int y = 0; y < ImgIn->hight; y++)
		{
			for (int x = 0; x < ImgIn->width; x++)
			{
				int globalPos = y * ImgIn->width + x;
				int midy = ImgIn->hight/2;
				if(y < midy)
				{
					pSrc[globalPos] = (hImage[(2 * y + 1) * ImgIn->width + x] + hImage[(2 * y) * ImgIn->width + x]) * INV_SQRT_2;
				}
				else
				{
					int neighborY = 2 * (y - midy) + 1;
					if(neighborY >= ImgIn->hight)
						neighborY = ImgIn->hight - 1;
					pSrc[globalPos] = (hImage[neighborY * ImgIn->width + x] - hImage[(2 * (y - midy)) * ImgIn->width + x]) * INV_SQRT_2;
				}
			}
		}
	}

	memcpy(ImgDst->GetData(false), pSrc, ImgIn->width * ImgIn->hight * sizeof(float));

	delete[] pSrc;
	delete[] hImage;
}

void Core::CGDwtHaar(CGImage<float> *ImgDst, CGImage<float> *ImgIn, int haar_level)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGDwtHaar_CUDA(ImgDst, ImgIn, haar_level);
#else
	CGDwtHaar_CPU(ImgDst, ImgIn, haar_level);
#endif
}

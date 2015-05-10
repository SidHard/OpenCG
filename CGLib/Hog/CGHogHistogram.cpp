//author 2015 Wang Xinbo

#include "CGHogHistogram.h"

using namespace CG;
using namespace CG::Hog;

void Hog::CGHogHistogram_CPU(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int cellSizeX, int cellSizeY, int windowSizeX, int windowSizeY)
{}

void Hog::CGHogHistogram(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int cellSizeX, int cellSizeY, int windowSizeX, int windowSizeY)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGHogHistogram_CUDA(hogHistogram, ImgGrad, ImgNorm, cellSizeX, cellSizeY, windowSizeX, windowSizeY);
#else
	CGHogHistogram_CPU(hogHistogram, ImgGrad, ImgNorm, cellSizeX, cellSizeY, windowSizeX, windowSizeY);
#endif
}
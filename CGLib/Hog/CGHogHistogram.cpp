//author 2015 Wang Xinbo

#include "CGHogHistogram.h"

using namespace CG;
using namespace CG::Hog;

void Hog::CGHogHistogram_CPU(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int hogCellSizeX, int hogCellSizeY)
{}

void Hog::CGHogHistogram(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int hogCellSizeX, int hogCellSizeY)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGHogHistogram_CUDA(hogHistogram, ImgGrad, ImgNorm, hogCellSizeX, hogCellSizeY);
#else
	CGHogHistogram_CPU(hogHistogram, ImgGrad, ImgNorm, hogCellSizeX, hogCellSizeY);
#endif
}
//author 2015 Wang Xinbo

#include "CGHogHistogram.h"

using namespace CG;
using namespace CG::Hog;

void Hog::CGHogHistogram_CPU(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm)
{}

void Hog::CGHogHistogram(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGHogHistogram_CUDA(hogHistogram, ImgGrad, ImgNorm);
#else
	CGHogHistogram_CPU(hogHistogram, ImgGrad, ImgNorm);
#endif
}
//author 2015 Wang Xinbo

#include "CGConvolution.h"

using namespace CG;
using namespace CG::Core;

void Core::CGComputeGradNorm_CPU(CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, CGImage<float> *ImgIn)
{}

void Core::CGComputeGradients_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn)
{
}

void Core::CGComputeGradient(CGImage<float> *ImgDst, CGImage<float> *ImgIn)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGComputeGradients_CUDA(ImgDst, ImgIn);
#else
	CGComputeGradients_CPU(ImgDst, ImgIn);
#endif
}

void Core::CGComputeGradNorm(CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, CGImage<float> *ImgIn)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGComputeGradNorm_CUDA(ImgGrad, ImgNorm, ImgIn);
#else
	CGComputeGradNorm_CPU(ImgGrad, ImgNorm, ImgIn);
#endif
}
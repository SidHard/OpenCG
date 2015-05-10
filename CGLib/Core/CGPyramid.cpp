//author 2015 Wang Xinbo

#include "CGPyramid.h"

using namespace CG;
using namespace CG::Core;

void Core::CGPyramid_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float scale)
{}

void Core::CGPyramid(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float scale)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGPyramid_CUDA(ImgDst, ImgIn, scale);
#else
	CGPyramid_CPU(ImgDst, ImgIn, scale);
#endif
}
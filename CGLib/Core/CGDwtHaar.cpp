//author 2015 Wang Xinbo

#include "CGDwtHaar.h"

using namespace CG;
using namespace CG::Core;

void Core::CGDwtHaar_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn)
{
}

void Core::CGDwtHaar(CGImage<float> *ImgDst, CGImage<float> *ImgIn)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGDwtHaar_CUDA(ImgDst, ImgIn);
#else
	CGDwtHaar_CPU(ImgDst, ImgIn);
#endif
}

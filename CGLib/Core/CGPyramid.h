//author 2015 Wang Xinbo

#pragma once

#include "CGImage.h"

namespace CG
{
	namespace Core
	{
		void CGPyramid(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float scale = 1);

		void CGPyramid_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float scale = 1);

#ifndef COMPILE_WITHOUT_CUDA
		void CGPyramid_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float scale = 1);
#endif
	}
}
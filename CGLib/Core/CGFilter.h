//author 2015 Wang Xinbo

#pragma once

#include "CGImage.h"

namespace CG
{
	namespace Core
	{
		void ConstructKernel_CPU(int radius, float delta = 4);

		void ConstructKernel_GPU(int radius, float delta = 4);

		void CGFilter_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float e_d = 1, int radius = 1, int iterations = 1);

		void CGFilter(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float e_d = 1, int radius = 1, int iterations = 1);

#ifndef COMPILE_WITHOUT_CUDA
		double CGFilter_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float e_d = 1, int radius = 1, int iterations = 1);
#endif
	}
}
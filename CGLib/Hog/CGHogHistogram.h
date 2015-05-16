//author 2015 Wang Xinbo
//
//hog µœ÷

#pragma once

#include "CGHogConfig.h"

using namespace CG::Core;

namespace CG
{
	namespace Hog
	{
		void CGHogHistogram(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm);

		void CGHogHistogram_CPU(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm);

#ifndef COMPILE_WITHOUT_CUDA
		__host__ void cgInitHistogram();

		__host__ void CGHogHistogram_CUDA(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm);
#endif
	}
}
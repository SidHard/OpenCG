//author 2015 Wang Xinbo
//
//hog µœ÷

#pragma once

#include "..\Core\CGCore.h"

using namespace CG::Core;

namespace CG
{
	namespace Hog
	{
		void CGHogHistogram(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int cellSizeX, int cellSizeY, int windowSizeX, int windowSizeY);

		void CGHogHistogram_CPU(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int cellSizeX, int cellSizeY, int windowSizeX, int windowSizeY);

#ifndef COMPILE_WITHOUT_CUDA
		void CGHogHistogram_CUDA(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int cellSizeX, int cellSizeY, int windowSizeX, int windowSizeY);
#endif
	}
}
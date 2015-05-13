//author 2015 Wang Xinbo
//
//hog µœ÷

#pragma once

#include "..\Core\CGCore.h"

#define BLOCK_SIZE_W 2
#define BLOCK_SIZE_H 2
#define NO_BINS 9

using namespace CG::Core;

namespace CG
{
	namespace Hog
	{
		void CGHogHistogram(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int hogCellSizeX = 4, int hogCellSizeY = 4);

		void CGHogHistogram_CPU(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int hogCellSizeX, int hogCellSizeY);

#ifndef COMPILE_WITHOUT_CUDA
		void CGHogHistogram_CUDA(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, int hogCellSizeX, int hogCellSizeY);
#endif
	}
}
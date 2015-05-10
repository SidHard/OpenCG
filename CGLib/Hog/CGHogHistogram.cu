//author 2015 Wang Xinbo

#include "CGHogHistogram.h"

#ifndef COMPILE_WITHOUT_CUDA

using namespace CG;
using namespace CG::Core;
using namespace CG::Hog;

__global__ void cgHogHistogram(float *hogHistogram, float *ImgGrad, float *ImgNorm, int noHistogramBins, 
							   int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY, int imgWidth, int imgHight)
{
}

__host__ void 
Hog::CGHogHistogram_CUDA(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, 
							int cellSizeX, int cellSizeY, int windowSizeX, int windowSizeY)
{
}

#endif
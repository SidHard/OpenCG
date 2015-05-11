//author 2015 Wang Xinbo

#include "CGHogHistogram.h"

#ifndef COMPILE_WITHOUT_CUDA

using namespace CG;
using namespace CG::Core;
using namespace CG::Hog;

texture<float, 1, cudaReadModeElementType> texGauss;
cudaArray* gaussArray;
cudaChannelFormatDesc channelDescGauss;

extern __shared__ float allShared[];

__global__ void cgHogHistogram(float *hogHistogram, float *ImgGrad, float *ImgNorm, int hogNoHistogramBins, 
							   int hogCellSizeX, int hogCellSizeY, int hogBlockSizeX, int hogBlockSizeY, int imgWidth, int imgHight)
{
}

__host__ void 
Hog::CGHogHistogram_CUDA(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, 
							int hogCellSizeX, int hogCellSizeY)
{
	int hogBlockSizeX = BLOCK_SIZE_W;
	int hogBlockSizeY = BLOCK_SIZE_H;
	int hogNoHistogramBins = NO_BINS;
	int noBlockX = ImgGrad->width / hogCellSizeX - hogBlockSizeX + 1;
	int noBlockY = ImgGrad->hight / hogCellSizeY - hogBlockSizeY + 1;

	dim3 blockSize(hogCellSizeX, hogBlockSizeX, hogBlockSizeY);
	dim3 gridSize(noBlockX, noBlockY);
	cgHogHistogram<<<gridSize, blockSize, hogNoHistogramBins * hogBlockSizeX * hogBlockSizeY * hogCellSizeX * hogBlockSizeX * hogBlockSizeY * sizeof(float)>>>
		(hogHistogram->GetData(true), ImgGrad->GetData(true), ImgNorm->GetData(true), hogNoHistogramBins, hogCellSizeX, hogCellSizeY, hogBlockSizeX, hogBlockSizeY, ImgGrad->width, ImgGrad->hight);
}

#endif
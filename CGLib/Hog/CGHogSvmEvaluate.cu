//author 2015 Wang Xinbo

#include "CGHogSvmEvaluate.h"

#ifndef COMPILE_WITHOUT_CUDA

using namespace CG;
using namespace CG::Core;
using namespace CG::Hog;

texture<float, 1, cudaReadModeElementType> texSvm;
cudaArray *svmArray = 0;
cudaChannelFormatDesc svmDesc;
float svmBias;

extern __shared__ float allSharedSvm[];

__host__ void Hog::cgInitSVM(float _svmBias, float* svmWeights, int svmWeightsCount)
{
	svmDesc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&svmArray, &svmDesc, svmWeightsCount, 1);
	cudaMemcpyToArray(svmArray, 0, 0, svmWeights, svmWeightsCount * sizeof(float), cudaMemcpyHostToDevice);
	svmBias = _svmBias;
	cudaBindTextureToArray(texSvm, svmArray, svmDesc);
}

__global__ void cgHogSvmEvaluate(float *svmScore, float svmBias, float *hogHistogram, int alignedBlockDimX, int stride, int hogWindowNumX, int hogWindowNumY, 
								 int hogNoHistogramBins, int hogBlockNumPerWindowX, int hogBlockNumPerWindowY, int hogCellSizeX, 
								 int hogCellSizeY, int hogBlockSizeX, int hogBlockSizeY, int scaleID, int imgWidth, int imgHight)
{
	float localValue = 0;
	float* localShared = (float*) allSharedSvm;

	int globalPos = threadIdx.x + blockIdx.x * hogNoHistogramBins * hogBlockSizeX + blockIdx.y * hogBlockSizeY * stride;
	int localPos = threadIdx.x;
	int targetPos;

	int val1 = (hogBlockSizeY * hogBlockSizeX * hogNoHistogramBins) * hogBlockNumPerWindowY;
	int val2 = hogBlockSizeX * hogNoHistogramBins;

	for (int i = 0; i<hogBlockSizeY * hogBlockNumPerWindowY; i++)
	{
		int iGlobalPos = globalPos + i * stride;
		int texPos = threadIdx.x % val2 + i * val2 + threadIdx.x / val2 * val1;
		float texValue =  tex1D(texSvm, texPos);
		localValue += hogHistogram[iGlobalPos] * texValue;

		////ต๗สิ
		//if(threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0)
		//{
		//	printf("*%.2f %f ", localValue, texValue);
		//}
	}

	localShared[localPos] = localValue;

	__syncthreads();

	for(unsigned int s = alignedBlockDimX >> 1; s>0; s>>=1)
	{
		if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x)
		{
			targetPos = threadIdx.x + s;
			localShared[localPos] += localShared[targetPos];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		localShared[localPos] -= svmBias;
		svmScore[blockIdx.x + blockIdx.y * hogWindowNumX + scaleID * hogWindowNumX * hogWindowNumY] = localShared[localPos];
	}

	////ต๗สิ
	//if(threadIdx.x == 0 && blockIdx.y < 10 && blockIdx.x < 10)
	//{
	//	printf("*%f ", localShared[localPos]);
	//}
}

__host__ void cgFreeSvm()
{
	cudaUnbindTexture(texSvm);
	cudaFreeArray(svmArray); 
}

__host__ void 
Hog::CGHogSvmEvaluate_CUDA(CGImage<float> *svmScore, CGImage<float> *hogHistogram, int imgWidth, int imgHight, int scaleId)
{
	int hogBlockSizeX = BLOCK_SIZE_W, hogBlockSizeY = BLOCK_SIZE_H;
	int hogCellSizeX = CELL_SIZE_W, hogCellSizeY = CELL_SIZE_H;
	int hogNoHistogramBins = NO_BINS;
	int hogWindowNumX = (imgWidth - WINDOW_SIZE_W)/hogCellSizeX + 1;
	int hogWindowNumY = (imgHight - WINDOW_SIZE_H)/hogCellSizeY + 1;
	int hogBlockNumPerWindowX = (WINDOW_SIZE_W - hogCellSizeX * hogBlockSizeX) / hogCellSizeX + 1;
	int hogBlockNumPerWindowY = (WINDOW_SIZE_H - hogCellSizeY * hogBlockSizeY) / hogCellSizeY + 1;
	int alignedBlockDimX = ClosestPowerOfTwo(hogNoHistogramBins * hogBlockSizeX * hogBlockNumPerWindowX);

	//cgInitSVM(_svmBias, svmWeights, svmWeightsCount);

	dim3 blockSize(hogNoHistogramBins * hogBlockSizeX * hogBlockNumPerWindowX);
	dim3 gridSize(hogWindowNumX, hogWindowNumY);
	cgHogSvmEvaluate<<<gridSize, blockSize, hogNoHistogramBins * hogBlockSizeX * hogWindowNumX * sizeof(float)>>>
		(svmScore->GetData(true), svmBias, hogHistogram->GetData(true), alignedBlockDimX, hogHistogram->width, hogWindowNumX, hogWindowNumY, hogNoHistogramBins,hogBlockNumPerWindowX, hogBlockNumPerWindowY, hogCellSizeX, hogCellSizeY, hogBlockSizeX, hogBlockSizeY, scaleId, imgWidth, imgHight);
}

#endif
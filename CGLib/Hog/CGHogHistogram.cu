//author 2015 Wang Xinbo
//要求window和image能整除cellsize

#include "CGHogHistogram.h"

#ifndef COMPILE_WITHOUT_CUDA

using namespace CG;
using namespace CG::Core;
using namespace CG::Hog;

__device__ __constant__ float centerBlock_pix[3], centerBlock[3], pixPerCell[3], oneHalf = 0.5f;
__device__ __constant__ int block_w_h_bins[3];
__device__ int d_sum = 0;

texture<float, 1, cudaReadModeElementType> texGauss;
cudaArray* gaussArray;
cudaChannelFormatDesc desc;

extern __shared__ float allShared[];

__host__ void cgHogInit(int hogCellSizeX, int hogCellSizeY)
{
	//变量初始化
	int h_block_w_h_bins[3];
	float h_centerBlock_pix[3], h_centerBlock[3], h_pixPerCell[3];

	h_centerBlock_pix[0] = hogCellSizeX * BLOCK_SIZE_W / 2.0f;  h_centerBlock_pix[1] = hogCellSizeY * BLOCK_SIZE_H / 2.0f;  h_centerBlock_pix[2] = 180 / 2.0f;
	h_centerBlock[0] = BLOCK_SIZE_W / 2.0f;						h_centerBlock[1] = BLOCK_SIZE_H / 2.0f;						h_centerBlock[2] = NO_BINS / 2.0f;
	h_pixPerCell[0] = (float) 1.0f / hogCellSizeX;				h_pixPerCell[1] = (float) 1.0f / hogCellSizeY;				h_pixPerCell[2] = (float) NO_BINS / 180.0f;
	h_block_w_h_bins[0] = BLOCK_SIZE_W;							h_block_w_h_bins[1] = BLOCK_SIZE_H;							h_block_w_h_bins[2] = NO_BINS;

	cudaMemcpyToSymbol(centerBlock_pix, h_centerBlock_pix, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(centerBlock, h_centerBlock, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(pixPerCell, h_pixPerCell, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(block_w_h_bins, h_block_w_h_bins, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);

	//gaussian核初始化
	float var2x = hogCellSizeX * BLOCK_SIZE_W / (2 * 2.0);//wt:gaussian参数
	float var2y = hogCellSizeY * BLOCK_SIZE_H / (2 * 2.0);//wt:gaussian参数
	var2x *= var2x * 2; var2y *= var2y * 2;

	float* weights = (float*)malloc(hogCellSizeX * BLOCK_SIZE_W * hogCellSizeY * BLOCK_SIZE_H * sizeof(float));

	for (int i=0; i<hogCellSizeX * BLOCK_SIZE_W; i++)
	{
		for (int j=0; j<hogCellSizeY * BLOCK_SIZE_H; j++)
		{
			float tx = i - h_centerBlock_pix[0];
			float ty = j - h_centerBlock_pix[1];
			tx *= tx / var2x;
			ty *= ty / var2y;
			weights[i + j * hogCellSizeX * BLOCK_SIZE_W] = exp(-(tx + ty));
		}
	}

	desc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&gaussArray, &desc, hogCellSizeX * BLOCK_SIZE_W * hogCellSizeY * BLOCK_SIZE_H, 1);
	cudaMemcpyToArray(gaussArray, 0, 0, weights, sizeof(float) * hogCellSizeX * BLOCK_SIZE_W * hogCellSizeY * BLOCK_SIZE_H, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texGauss, gaussArray, desc);

	//释放指针
	//free(weights); 
	//delete[] h_centerBlock_pix; delete[] h_centerBlock; delete[] h_pixPerCell; delete[] h_block_w_h_bins;
}

__global__ void cgHogHistogram(float *hogHistogram, float *ImgGrad, float *ImgNorm, int hogNoHistogramBins, 
							   int hogCellSizeX, int hogCellSizeY, int hogBlockSizeX, int hogBlockSizeY, int imgWidth, int imgHight)
{
	float2 localValue;
	float* localShared = (float*)allShared;

	int cellIdx = threadIdx.y;
	int cellIdy = threadIdx.z;
	int columnId = threadIdx.x;

	//histogram list坐标
	int localCellBinPos = cellIdx * hogNoHistogramBins + cellIdy * hogBlockSizeX * hogNoHistogramBins;
	int globalCellBinPos = cellIdx * hogNoHistogramBins + cellIdy * gridDim.x * hogBlockSizeX * hogNoHistogramBins +
		blockIdx.x * hogNoHistogramBins * hogBlockSizeX + blockIdx.y * gridDim.x * hogBlockSizeX * hogNoHistogramBins * hogBlockSizeY;

	//线程起始点坐标，每个线程hogCellSizeX个pix
	int globalPos = (blockIdx.x * hogCellSizeX + blockIdx.y * hogCellSizeY * imgWidth) + columnId + cellIdx * hogCellSizeX + cellIdy * hogCellSizeY * imgWidth;
	int localPos = columnId + cellIdx * hogCellSizeX + cellIdy * hogCellSizeY * hogCellSizeX * hogBlockSizeX;

	int histogramSize = hogNoHistogramBins * hogBlockSizeX * hogBlockSizeY;
	//线程局部histogram list坐标
	int histogramPos = (columnId + cellIdx * hogCellSizeX + cellIdy * hogBlockSizeX * hogCellSizeX) * histogramSize;
	for (int i=0; i<histogramSize; i++) localShared[histogramPos + i] = 0;  //init shared memory

	float atx, aty;
	float pIx, pIy, pIz;
	int fIx, fIy, fIz;
	int cIx, cIy, cIz;
	float dx, dy, dz;
	float cx, cy, cz;

	bool lowervalidx, lowervalidy;
	bool uppervalidx, uppervalidy;
	bool canWrite;

	int offset;

	//localShared初始化
	for (int i=0; i<hogCellSizeY; i++)
	{
		localValue.x = ImgGrad[globalPos + i * imgWidth];
		localValue.y = ImgNorm[globalPos + i * imgWidth];
		localValue.x *= tex1D(texGauss, localPos + i * hogCellSizeX * hogBlockSizeX);

		//if (blockIdx.x == 2 && blockIdx.y == 2 && threadIdx.x == 0)
		//{
		//	printf("#PIX%d %d %.3f %.3f", threadIdx.y, threadIdx.z, localValue.x, localValue.y);
		//}

		atx = cellIdx * hogCellSizeX + columnId + 0.5;
		aty = cellIdy * hogCellSizeY + i + 0.5;

		pIx = centerBlock[0] - 0.5f + (atx - centerBlock_pix[0]) * pixPerCell[0];
		pIy = centerBlock[1] - 0.5f + (aty - centerBlock_pix[1]) * pixPerCell[1];
		pIz = centerBlock[2] - 0.5f + (localValue.y - centerBlock_pix[2]) * pixPerCell[2];

		fIx = floorf(pIx); fIy = floorf(pIy); fIz = floorf(pIz);
		cIx = fIx + 1; cIy = fIy + 1; cIz = fIz + 1; 

		dx = pIx - fIx; dy = pIy - fIy; dz = pIz - fIz;
		cx = 1 - dx; cy = 1 - dy; cz = 1 - dz;

		cIz %= hogNoHistogramBins;
		fIz %= hogNoHistogramBins;
		if (fIz < 0) fIz += hogNoHistogramBins;
		if (cIz < 0) cIz += hogNoHistogramBins;

		uppervalidx = !(cIx >= hogBlockSizeX - oneHalf || cIx < -oneHalf);
		uppervalidy = !(cIy >= hogBlockSizeY - oneHalf || cIy < -oneHalf);
		lowervalidx = !(fIx < -oneHalf || fIx >= hogBlockSizeX - oneHalf);
		lowervalidy = !(fIy < -oneHalf || fIy >= hogBlockSizeY - oneHalf);

		canWrite = (lowervalidx) && (lowervalidy);
		if (canWrite)
		{
			offset = histogramPos + (fIx + fIy * hogBlockSizeY) * hogNoHistogramBins;
			localShared[offset + fIz] += localValue.x * cx * cy * cz;
			localShared[offset + cIz] += localValue.x * cx * cy * dz;
		}

		canWrite = (lowervalidx) && (uppervalidy);
		if (canWrite)
		{
			offset = histogramPos + (fIx + cIy * hogBlockSizeY) * hogNoHistogramBins;
			localShared[offset + fIz] += localValue.x * cx * dy * cz;
			localShared[offset + cIz] += localValue.x * cx * dy * dz;
		}

		canWrite = (uppervalidx) && (lowervalidy);
		if (canWrite)
		{
			offset = histogramPos + (cIx + fIy * hogBlockSizeY) * hogNoHistogramBins;
			localShared[offset + fIz] += localValue.x * dx * cy * cz;
			localShared[offset + cIz] += localValue.x * dx * cy * dz;
		}

		canWrite = (uppervalidx) && (uppervalidy);
		if (canWrite)
		{
			offset = histogramPos + (cIx + cIy * hogBlockSizeY) * hogNoHistogramBins;
			localShared[offset + fIz] += localValue.x * dx * dy * cz;
			localShared[offset + cIz] += localValue.x * dx * dy * dz;
			//if (blockIdx.x == 10 && blockIdx.y == 10)
			//{
			//	printf("#%d %d", offset + fIz, offset + cIz);
			//}
		}
	}

	__syncthreads();

	//if (blockIdx.x == 5 && blockIdx.y == 2)
	//{
	//	printf("*%d %.5f %.5f", localPos, localShared[histogramPos], localShared[histogramPos + 1]);
	//}

	//block内的localShared加到前36个数中，分别代表四个cell中的bins
	int targetHistogramPos;
	for(unsigned int s = hogBlockSizeY >> 1; s>0; s>>=1)
	{
		if (cellIdy < s && (cellIdy + s) < hogBlockSizeY)
		{
			targetHistogramPos = (columnId + cellIdx * hogCellSizeX) * histogramSize + (cellIdy + s) * histogramSize * hogBlockSizeX * hogCellSizeX;

			for (int i=0; i<histogramSize; i++)
				localShared[histogramPos + i] += localShared[targetHistogramPos + i];
		}

		__syncthreads();
	}

	for(unsigned int s = hogBlockSizeX >> 1; s>0; s>>=1)
	{
		if (cellIdx < s && (cellIdx + s) < hogBlockSizeX)
		{
			targetHistogramPos = (columnId + (cellIdx + s) * hogCellSizeX) * histogramSize + cellIdy * histogramSize * hogBlockSizeX * hogCellSizeX;

			for (int i=0; i<histogramSize; i++)
				localShared[histogramPos + i] += localShared[targetHistogramPos + i];
		}

		__syncthreads();
	}

	for(unsigned int s = hogCellSizeX >> 1; s>0; s>>=1)
	{
		if (columnId < s && (columnId + s) < hogCellSizeX)
		{
			targetHistogramPos = (columnId + s + cellIdx * hogCellSizeX) * histogramSize + cellIdy * histogramSize * hogBlockSizeX * hogCellSizeX;

			for (int i=0; i<histogramSize; i++)
				localShared[histogramPos + i] += localShared[targetHistogramPos + i];
		}

		__syncthreads();
	}

	//输出，每个CELL有9维BIN向量
	if (columnId == 0)
	{
		for (int i=0; i<hogNoHistogramBins; i++)
			hogHistogram[globalCellBinPos + i] = localShared[localCellBinPos + i];

		//if (blockIdx.x < 10 && blockIdx.y < 10)
		//{
		//	printf("#%d %d %.3f", blockIdx.x, blockIdx.y, localShared[localCellBinPos]);
		//}
	}
}

__global__ void normalizeHistograms(float *blockHistograms, int hogNoHistogramBins, int noBlockX, int noBlockY, int hogBlockSizeX, int hogBlockSizeY,
										 int alignedBlockDimX, int alignedBlockDimY, int alignedBlockDimZ, int imgBinW, int imgBinH)
{
	float* localShared = (float*)allShared;

	float localValue, norm1, norm2; float eps2 = 0.01f;

	int cellIdx = threadIdx.y;
	int cellIdy = threadIdx.z;
	int binId = threadIdx.x;

	int targetPos;
	int localPos = cellIdx * hogNoHistogramBins + cellIdy * hogNoHistogramBins * blockDim.y + threadIdx.x;
	int globalPos = blockIdx.y * gridDim.x * blockDim.y * hogNoHistogramBins * blockDim.z + cellIdy * gridDim.x * blockDim.y * hogNoHistogramBins
		+ blockIdx.x * hogNoHistogramBins * blockDim.y + cellIdx * hogNoHistogramBins + threadIdx.x;
	//gmemWritePosBlock = cellIdy * hogNoHistogramBins + cellIdx * gridDim.x * blockDim.y * hogNoHistogramBins +
	//	threadIdx.x + blockIdx.x * hogNoHistogramBins * blockDim.y + blockIdx.y * gridDim.x * blockDim.y * hogNoHistogramBins * blockDim.z;

	localValue = blockHistograms[globalPos];
	localShared[localPos] = localValue * localValue;

	__syncthreads();

	for(unsigned int s = alignedBlockDimZ >> 1; s>0; s>>=1)
	{
		if (threadIdx.z < s && (threadIdx.z + s) < blockDim.z)
		{
			targetPos = threadIdx.y * hogNoHistogramBins + (threadIdx.z + s) * blockDim.x * blockDim.y + threadIdx.x;
			localShared[localPos] += localShared[targetPos];
		}

		__syncthreads();

	}

	for (unsigned int s = alignedBlockDimY >> 1; s>0; s>>=1)
	{
		if (threadIdx.y < s && (threadIdx.y + s) < blockDim.y)
		{
			targetPos = (threadIdx.y + s) * hogNoHistogramBins + threadIdx.z * blockDim.x * blockDim.y + threadIdx.x;
			localShared[localPos] += localShared[targetPos];
		}

		__syncthreads();

	}

	for(unsigned int s = alignedBlockDimX >> 1; s>0; s>>=1)
	{
		if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x)
		{
			targetPos = threadIdx.y * hogNoHistogramBins + threadIdx.z * blockDim.x * blockDim.y + (threadIdx.x + s);
			localShared[localPos] += localShared[targetPos];
		}

		__syncthreads();
	}

	norm1 = sqrtf(localShared[0]) + hogNoHistogramBins * hogBlockSizeX * hogBlockSizeY;
	localValue /= norm1;

	localValue = fminf(0.2f, localValue); 

	__syncthreads();

	localShared[localPos] = localValue * localValue;

	__syncthreads();

	for(unsigned int s = alignedBlockDimZ >> 1; s>0; s>>=1)
	{
		if (threadIdx.z < s && (threadIdx.z + s) < blockDim.z)
		{
			targetPos = threadIdx.y * hogNoHistogramBins + (threadIdx.z + s) * blockDim.x * blockDim.y + threadIdx.x;
			localShared[localPos] += localShared[targetPos];
		}

		__syncthreads();

	}

	for (unsigned int s = alignedBlockDimY >> 1; s>0; s>>=1)
	{
		if (threadIdx.y < s && (threadIdx.y + s) < blockDim.y)
		{
			targetPos = (threadIdx.y + s) * hogNoHistogramBins + threadIdx.z * blockDim.x * blockDim.y + threadIdx.x;
			localShared[localPos] += localShared[targetPos];
		}

		__syncthreads();

	}

	for(unsigned int s = alignedBlockDimX >> 1; s>0; s>>=1)
	{
		if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x)
		{
			targetPos = threadIdx.y * hogNoHistogramBins + threadIdx.z * blockDim.x * blockDim.y + (threadIdx.x + s);
			localShared[localPos] += localShared[targetPos];
		}

		__syncthreads();
	}

	norm2 = sqrtf(localShared[0]) + eps2;
	localValue /= norm2;

	blockHistograms[globalPos] = localValue;
}

__host__ void cgHogFree()
{
	cudaUnbindTexture(texGauss);
	cudaFree(centerBlock_pix); cudaFree(centerBlock); cudaFree(pixPerCell); cudaFree(block_w_h_bins); cudaFree(gaussArray);
}

__host__ void 
Hog::CGHogHistogram_CUDA(CGImage<float> *hogHistogram, CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, 
							int hogCellSizeX, int hogCellSizeY)
{
	int hogBlockSizeX = BLOCK_SIZE_W;
	int hogBlockSizeY = BLOCK_SIZE_H;
	int hogNoHistogramBins = NO_BINS;
	int noCellX = ImgGrad->width / hogCellSizeX;
	int noCellY = ImgGrad->hight / hogCellSizeY;
	int noBlockX = ImgGrad->width / hogCellSizeX - hogBlockSizeX + 1;
	int noBlockY = ImgGrad->hight / hogCellSizeY - hogBlockSizeY + 1;

	cgHogInit(hogCellSizeX, hogCellSizeY);

	dim3 blockSize(hogCellSizeX, hogBlockSizeX, hogBlockSizeY);
	dim3 gridSize(noBlockX, noBlockY);
	cgHogHistogram<<<gridSize, blockSize, hogNoHistogramBins * hogBlockSizeX * hogBlockSizeY * hogCellSizeX * hogBlockSizeX * hogBlockSizeY * sizeof(float)>>>
		(hogHistogram->GetData(true), ImgGrad->GetData(true), ImgNorm->GetData(true), hogNoHistogramBins, hogCellSizeX, hogCellSizeY, hogBlockSizeX, hogBlockSizeY, ImgGrad->width, ImgGrad->hight);

	int alignedBlockDimX = 16;
	int alignedBlockDimY = 2;
	int alignedBlockDimZ = 2;
	blockSize = dim3(hogNoHistogramBins, hogBlockSizeX, hogBlockSizeY);
	gridSize = dim3(noBlockX, noBlockY);
	normalizeHistograms<<<gridSize, blockSize, hogNoHistogramBins * hogBlockSizeX * hogBlockSizeY * sizeof(float)>>>
		(hogHistogram->GetData(true), hogNoHistogramBins, noBlockX, noBlockY, hogBlockSizeX, hogBlockSizeY, alignedBlockDimX, alignedBlockDimY, alignedBlockDimZ, hogNoHistogramBins * noCellX, noCellY);

	cgHogFree();
}

#endif
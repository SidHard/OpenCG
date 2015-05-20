//author 2015 Wang Xinbo

#include "CGDwtHaar.h"

#ifndef COMPILE_WITHOUT_CUDA

using namespace CG;
using namespace CG::Core;

texture<float, 2, cudaReadModeElementType> texHaar;

//__global__ void CGConvRow(unsigned char *d_Result, int dataW, int dataH)
__global__ void CGHaarRow(float *d_Result, int dataW, int dataH, int midSize, int stride)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int globalPos = y * stride + x;
	float norm = INV_SQRT_2;

	if(x<dataW && y<dataH)
	{
		if(x < midSize)
		{
			d_Result[globalPos] = (tex2D(texHaar, 2 * x + 1, y) + tex2D(texHaar, 2 * x, y)) * norm;
		}
		else
		{
			d_Result[globalPos] = (tex2D(texHaar, 2 * (x - midSize) + 1, y) - tex2D(texHaar, 2 * (x - midSize), y)) * norm;
		}
	}
	////调试
	//if(0 == globalPos%1000)
	//{
	//	printf("*%d %.2f", globalPos, d_Result[globalPos]);
	//}

}

//Y轴卷积
__global__ void CGHaarColumn ( float *d_Result, int dataW, int dataH, int midSize, int stride)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int globalPos = y * stride + x;
	float norm = INV_SQRT_2;

	if(x<dataW && y<dataH)
	{
		if(y < midSize)
		{
			d_Result[globalPos] = (tex2D(texHaar, x, 2 * y + 1) + tex2D(texHaar, x, 2 * y)) * norm;
		}
		else
		{
			d_Result[globalPos] = (tex2D(texHaar, x, 2 * (y - midSize) + 1) - tex2D(texHaar, x, 2 * (y - midSize))) * norm;
		}
	}
}

__host__ void 
Core::CGDwtHaar_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn, int haar_level)
{
	dim3 gridSize((ImgIn->width + 16 - 1) / 16, (ImgIn->hight + 16 - 1) / 16);
	dim3 blockSize(16, 16);

	//绑定纹理
	const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	texHaar.normalized = false;
	float *haarBufferRow;
	float *haarBufferCol;
	size_t pitch;
	cudaMallocPitch((void**)&haarBufferRow, &pitch, ImgIn->width*sizeof(float), ImgIn->hight);
	cudaMallocPitch((void**)&haarBufferCol, &pitch, ImgIn->width*sizeof(float), ImgIn->hight);
	cudaMemcpy2D(haarBufferRow, pitch, ImgIn->GetData(true), ImgIn->width*sizeof(float), ImgIn->width*sizeof(float), ImgIn->hight, cudaMemcpyDeviceToDevice);

	int stride = pitch/sizeof(float);

	for(int i = 0; i < haar_level; i++)
	{
		cudaBindTexture2D(0, texHaar, haarBufferRow, desc, ImgIn->width, ImgIn->hight, pitch);

		CGHaarRow<<<gridSize, blockSize>>>(haarBufferCol, ImgIn->width, ImgIn->hight, ImgIn->width/2, stride);

		cudaBindTexture2D(0, texHaar, haarBufferCol, desc, ImgIn->width, ImgIn->hight, pitch);

		CGHaarColumn<<<gridSize, blockSize>>>(haarBufferRow, ImgIn->width, ImgIn->hight, ImgIn->hight/2, stride);
	}

	cudaMemcpy2D(ImgDst->GetData(true), ImgIn->width*sizeof(float), haarBufferRow, pitch, ImgIn->width*sizeof(float), ImgIn->hight, cudaMemcpyDeviceToDevice);


	cudaFree(haarBufferRow);
	cudaFree(haarBufferCol);
	cudaUnbindTexture(texHaar);
}

#endif
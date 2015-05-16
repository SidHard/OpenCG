//author 2015 Wang Xinbo

#include "CGPyramid.h"

#ifndef COMPILE_WITHOUT_CUDA

using namespace CG;
using namespace CG::Core;

texture<float, 2, cudaReadModeElementType> tex;

__global__ void fastResize(float *result, int width, int hight, int dstWidth, int dstHight, float scale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int globalPos = y * width + x;

	float u = x*scale;
	float v = y*scale;

	if(x<width && y<hight && x<dstWidth && y<dstHight)
		result[globalPos] = tex2D(tex, u, v);

	//if(x<20 && y<10)
	//{
	//	printf("*%.2f %d ", result[globalPos], x);
	//}
}

__host__ void 
Core::CGPyramid_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float scale)
{
	int newWidth = (int)(ImgIn->width/scale);
	int newHight = (int)(ImgIn->hight/scale);
	dim3 gridSize((newWidth + 16 - 1) / 16, (newHight + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	//∞Û∂®Œ∆¿Ì
	const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false;
	float *inTex;
	size_t pitch;
	cudaMallocPitch((void**)&inTex, &pitch, ImgIn->width*sizeof(float), ImgIn->hight);
	cudaMemcpy2D(inTex, pitch, ImgIn->GetData(true), ImgIn->width*sizeof(float), ImgIn->width*sizeof(float), ImgIn->hight, cudaMemcpyDeviceToDevice);
	cudaBindTexture2D(0, tex, inTex, desc, ImgIn->width, ImgIn->hight, pitch);

	ImgDst->Clear();
	//ImgDst->ChangeSize(newWidth, newHight);
	fastResize<<<gridSize, blockSize>>>(ImgDst->GetData(true), ImgDst->width, ImgDst->hight, newWidth, newHight, scale);

	cudaUnbindTexture(tex);
	cudaFree(inTex);
}

#endif
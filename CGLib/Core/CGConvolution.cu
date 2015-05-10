//author 2015 Wang Xinbo

#include "CGConvolution.h"

#ifndef COMPILE_WITHOUT_CUDA

using namespace CG;
using namespace CG::Core;

#define convKernelRadius 1
#define convKernelWidth (2 * convKernelRadius + 1)
__device__ __constant__ float d_Kernel[convKernelWidth];
texture<float, 1, cudaReadModeElementType> tex;

#define convRowTileWidth 128
#define convColumnTileWidth 16
#define convColumnTileHeight 48

#ifndef RADTODEG
#define RADTODEG 57.2957795
#endif

float1 *convBuffer;

//X,Y轴卷积核函数，递归调用
template<int i> __device__ float1 convolutionRow(float1 *data) {
	float1 val = data[convKernelRadius-i];
	val.x *= d_Kernel[i];
	val.x += convolutionRow<i-1>(data).x;
	return val;
}
template<> __device__ float1 convolutionRow<-1>(float1 *data){float1 zero; zero.x = 0; return zero;}
template<int i> __device__ float1 convolutionColumn(float1 *data) {
	float1 val = data[(convKernelRadius-i)*convColumnTileWidth];
	val.x *= d_Kernel[i];
	val.x += convolutionColumn<i-1>(data).x;
	return val;
}
template<> __device__ float1 convolutionColumn<-1>(float1 *data){float1 zero; zero.x = 0; return zero;}

///////////////////////////////////////////////////////
//               X轴卷积
//
//    共享内存为待卷积行左右各扩一个像素
//
//////////////////////////////////////////////////////

//__global__ void CGConvRow(unsigned char *d_Result, int dataW, int dataH)
__global__ void CGConvRow(float1 *d_Result, int dataW, int dataH)
{
	const int globalPos = blockIdx.y * dataW + blockIdx.x * convRowTileWidth + threadIdx.x - convKernelRadius;
	//const int rowPos = blockIdx.x * convRowTileWidth + threadIdx.x - convKernelRadius;
	const int localPos = threadIdx.x;
	//float1 zero; zero.x = 0;

	__shared__ float1 data[convKernelRadius + convRowTileWidth + convKernelRadius];

	float1 pix;
	pix.x= tex1Dfetch(tex, globalPos);
	//data[localPos] = ((rowPos >= 0) && (rowPos <= dataW - 1)) ? pix : zero;
	data[localPos] = pix;

	__syncthreads();

	if(localPos <= convRowTileWidth + convKernelRadius - 1  && localPos >= convKernelRadius)
	{
		float1 sum = convolutionRow<2 * convKernelRadius>(data + localPos);
		d_Result[globalPos] = sum;

		////调试
		//if(0 == globalPos%1000)
		//{
		//	printf("*%d %d %.2f %.2f %.2f", globalPos, localPos, pix.x, sum.x, data[localPos].x);
		//}
		//d_Result[globalPos] = floatToUchar(sum.x);
		////
	}
}

//__global__ void CGConvColumn ( unsigned char *d_Result, float1 *d_DataRow, int dataW, int dataH, int localStride, int globalStride)
__global__ void CGConvColumn_f ( float *d_Grad, float *d_Norm, float1 *d_DataRow, int dataW, int dataH, int localStride, int globalStride)
{
	float1 rowValue;
	float1 zero; zero.x = 0;
	float2 result;

	int globalPos = (blockIdx.y * convColumnTileHeight + threadIdx.y - convKernelRadius) * dataW + blockIdx.x * convColumnTileWidth + threadIdx.x;
	int localPos = threadIdx.y * convColumnTileWidth + threadIdx.x;

	__shared__ float1 data[convColumnTileWidth * (convKernelRadius + convColumnTileHeight + convKernelRadius)];

	for(int y = blockIdx.y * convColumnTileHeight + threadIdx.y; y <= (blockIdx.y + 1) * convColumnTileHeight + 2 * convKernelRadius - 1; y += blockDim.y)
	{
		float1 pix;
		pix.x= (((y-convKernelRadius) >= 0) && ((y-convKernelRadius) <= dataH - 1)) ? tex1Dfetch(tex, globalPos) : 0;
		data[localPos] =  pix;
		localPos += localStride;
		globalPos += globalStride;

		////调试
		//if(0 == threadIdx.y && blockIdx.y == 0)
		//{
		//	printf("*%d %d %.2f %.2f", globalPos, localPos, pix.x, data[localPos].x);
		//}
	}

	__syncthreads();

	localPos = (threadIdx.y + convKernelRadius) * convColumnTileWidth + threadIdx.x;
	globalPos = (blockIdx.y * convColumnTileHeight + threadIdx.y) * dataW + blockIdx.x * convColumnTileWidth + threadIdx.x;

	for(int y = blockIdx.y * convColumnTileHeight + threadIdx.y; y <= (blockIdx.y + 1) * convColumnTileHeight + threadIdx.y - 1; y += blockDim.y)
	{
		float1 sum = convolutionColumn<2 * convKernelRadius>(data + localPos);

		rowValue = d_DataRow[globalPos];
		result.x = sqrtf(sum.x * sum.x + rowValue.x * rowValue.x);
		result.y = atan2f(sum.x, rowValue.x) * RADTODEG;                 //弧度转角度
		d_Grad[globalPos] = result.x;
		d_Norm[globalPos] = result.y;
		//d_Result[globalPos] = unsigned char(__saturatef(fabs(result.x))*255.0f);

		localPos += localStride;
		globalPos += globalStride;
	}
}

//Y轴卷积
__global__ void CGConvColumn ( float *d_Result, float1 *d_DataRow, int dataW, int dataH, int localStride, int globalStride)
{
	float1 rowValue;
	float1 zero; zero.x = 0;
	float2 result;

	int globalPos = (blockIdx.y * convColumnTileHeight + threadIdx.y - convKernelRadius) * dataW + blockIdx.x * convColumnTileWidth + threadIdx.x;
	int localPos = threadIdx.y * convColumnTileWidth + threadIdx.x;

	__shared__ float1 data[convColumnTileWidth * (convKernelRadius + convColumnTileHeight + convKernelRadius)];

	for(int y = blockIdx.y * convColumnTileHeight + threadIdx.y; y <= (blockIdx.y + 1) * convColumnTileHeight + 2 * convKernelRadius - 1; y += blockDim.y)
	{
		float1 pix;
		pix.x= (((y-convKernelRadius) >= 0) && ((y-convKernelRadius) <= dataH - 1)) ? tex1Dfetch(tex, globalPos) : 0;
		data[localPos] =  pix;
		localPos += localStride;
		globalPos += globalStride;

		////调试
		//if(0 == threadIdx.y && blockIdx.y == 0)
		//{
		//	printf("*%d %d %.2f %.2f", globalPos, localPos, pix.x, data[localPos].x);
		//}
	}

	__syncthreads();

	localPos = (threadIdx.y + convKernelRadius) * convColumnTileWidth + threadIdx.x;
	globalPos = (blockIdx.y * convColumnTileHeight + threadIdx.y) * dataW + blockIdx.x * convColumnTileWidth + threadIdx.x;

	for(int y = blockIdx.y * convColumnTileHeight + threadIdx.y; y <= (blockIdx.y + 1) * convColumnTileHeight + threadIdx.y - 1; y += blockDim.y)
	{
		float1 sum = convolutionColumn<2 * convKernelRadius>(data + localPos);

		rowValue = d_DataRow[globalPos];
		result.x = sqrtf(sum.x * sum.x + rowValue.x * rowValue.x);
		result.y = atan2f(sum.x, rowValue.x) * RADTODEG;                 //弧度转角度
		d_Result[globalPos] = __saturatef(fabs(result.x));

		localPos += localStride;
		globalPos += globalStride;
	}
}

__host__ void 
Core::CGComputeGradients_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn)
{
	dim3 GridRows = dim3((ImgIn->width + convRowTileWidth - 1) / convRowTileWidth, ImgIn->hight);
	dim3 GridColumns = dim3((ImgIn->width + convColumnTileWidth - 1) / convColumnTileWidth, (ImgIn->hight + convColumnTileHeight - 1) / convColumnTileHeight);
	dim3 BlockRows = dim3(convKernelRadius + convRowTileWidth + convKernelRadius);
	dim3 BlockColumns = dim3(convColumnTileWidth, 8);

	//准备Kernel
	float *h_Kernel;
	h_Kernel = (float *)malloc(convKernelWidth * sizeof(float));
	h_Kernel[0] = 1.0f; h_Kernel[1] = 0;  h_Kernel[2] = -1.0f;  //h_Kernel[3] = 0;  h_Kernel[4] = 0; h_Kernel[5] = 0; h_Kernel[6] = -1.0f;
	cudaMemcpyToSymbol(d_Kernel, h_Kernel, convKernelWidth * sizeof(float));

	//绑定纹理
	const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture(0, tex, ImgIn->GetData(true), desc, ImgIn->width*ImgIn->hight*sizeof(float));

	cudaMalloc((void**) &convBuffer, sizeof(float1) * ImgIn->width * ImgIn->hight);

	CGConvRow<<<GridRows, BlockRows>>>(convBuffer, ImgIn->width, ImgIn->hight);
	//CGConvRow<<<GridRows, BlockRows>>>(ImgDst->GetData(true), ImgIn->width, ImgIn->hight);
	//CGConvColumn<<<GridColumns, BlockColumns>>>(outputImage, convBuffer, ImgIn->width, ImgIn->hight, convColumnTileWidth * BlockColumns.y, ImgIn->width * BlockColumns.y);
	CGConvColumn<<<GridColumns, BlockColumns>>>(ImgDst->GetData(true), convBuffer, ImgIn->width, ImgIn->hight, convColumnTileWidth * BlockColumns.y, ImgIn->width * BlockColumns.y);

	free(h_Kernel);
	cudaFree(convBuffer);
	cudaUnbindTexture(tex);
}

__host__ void 
Core::CGComputeGradNorm_CUDA(CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, CGImage<float> *ImgIn)
{
	dim3 GridRows = dim3((ImgIn->width + convRowTileWidth - 1) / convRowTileWidth, ImgIn->hight);
	dim3 GridColumns = dim3((ImgIn->width + convColumnTileWidth - 1) / convColumnTileWidth, (ImgIn->hight + convColumnTileHeight - 1) / convColumnTileHeight);
	dim3 BlockRows = dim3(convKernelRadius + convRowTileWidth + convKernelRadius);
	dim3 BlockColumns = dim3(convColumnTileWidth, 8);

	//准备Kernel
	float *h_Kernel;
	h_Kernel = (float *)malloc(convKernelWidth * sizeof(float));
	h_Kernel[0] = 1.0f; h_Kernel[1] = 0;  h_Kernel[2] = -1.0f;  //h_Kernel[3] = 0;  h_Kernel[4] = 0; h_Kernel[5] = 0; h_Kernel[6] = -1.0f;
	cudaMemcpyToSymbol(d_Kernel, h_Kernel, convKernelWidth * sizeof(float));

	//绑定纹理
	const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture(0, tex, ImgIn->GetData(true), desc, ImgIn->width*ImgIn->hight*sizeof(float));

	cudaMalloc((void**) &convBuffer, sizeof(float1) * ImgIn->width * ImgIn->hight);

	CGConvRow<<<GridRows, BlockRows>>>(convBuffer, ImgIn->width, ImgIn->hight);
	CGConvColumn_f<<<GridColumns, BlockColumns>>>(ImgGrad->GetData(true), ImgNorm->GetData(true), convBuffer, ImgIn->width, ImgIn->hight, convColumnTileWidth * BlockColumns.y, ImgIn->width * BlockColumns.y);

	free(h_Kernel);
	cudaFree(convBuffer);
	cudaUnbindTexture(tex);
}

#endif
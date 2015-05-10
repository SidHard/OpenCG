//author 2015 Wang Xinbo

#include "CGFilter.h"
//#include "opencv.hpp"

#ifndef COMPILE_WITHOUT_CUDA

//using namespace cv;
using namespace CG;
using namespace CG::Core;

__constant__ float GaussianKernel[64];   //GPU高斯核cudaReadModeElementType
texture<float, 1, cudaReadModeElementType> tex;
//texture<unsigned char, 1, cudaReadModeElementType> ucharTex;

//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
}

__device__ float euclideanLenUchar(float a, float b, float d)
{

    float mod = (b - a) * (b - a);

    return __expf(-mod / (2.f * d * d));
}

__global__ void
d_bilateral_filter(float *od, int w, int h, float e_d,  int r)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float t = {0.f};
    float center = tex1Dfetch(tex, x + y * w);

    for (int i = -r; i <= r; i++)
    {
        for (int j = -r; j <= r; j++)
        {
			float curPix = (x+j>0 && x+j<w && y+i>0 && y+i<h) ? tex1Dfetch(tex, x + j + (y + i)*w) : 0;
			factor = GaussianKernel[i + r] * GaussianKernel[j + r] *     //
						euclideanLenUchar(curPix, center, e_d);             //

			t += factor * curPix;
			sum += factor;
        }
    }

	////调试
	//if(0 == x%100 && 0 == y%100)
	//{
	//	printf("*%d %d %f %f", x, y, center, t);
	//}
	////

    od[y * w + x] = __saturatef(fabs(t/sum));   //floatToUchar
}

__host__ void
Core::ConstructKernel_GPU(int radius, float delta)
{
    float  fGaussian[64];

    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta));
    }

    cudaMemcpyToSymbol(GaussianKernel, fGaussian, sizeof(float)*(2*radius+1));
}

__host__ double 
Core::CGFilter_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn, float e_d, int radius, int iterations)
{
	//StopWatchInterface *timer;
 //   double dKernelTime;
	CGImage<float> *dTemp = new CGImage<float>(ImgIn->width, ImgIn->hight, true);
	dTemp->SetFrom(ImgIn, false, true);

	ConstructKernel_GPU(radius);

	////调试
	//dTemp->UpdateHostFromDevice();
	//Mat dispImg(480, 640, CV_32FC1, dTemp->GetData(false));
	//imshow("3", dispImg);
	////

    //绑定纹理
	const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture(0, tex, dTemp->GetData(true), desc, ImgIn->width*ImgIn->hight*sizeof(float));

    for (int i=0; i<iterations; i++)
    {
        // 计算每次滤波时间
        //dKernelTime = 0.0;
        cudaDeviceSynchronize();
        //sdkResetTimer(&timer);

		dim3 gridSize((ImgIn->width + 16 - 1) / 16, (ImgIn->hight + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_bilateral_filter<<< gridSize, blockSize>>>(ImgDst->GetData(true), ImgIn->width, ImgIn->hight, e_d, radius);

        // 
        cudaDeviceSynchronize();
        //dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // 迭代次数大于1，每次迭代都需要更新dTemp
			dTemp->SetFrom(ImgDst, false, true);
			cudaBindTexture(0, tex, dTemp->GetData(true), desc, ImgIn->width*ImgIn->hight);
        }
    }

	dTemp->Free();
	cudaUnbindTexture(tex);

    //return ((dKernelTime/1000.)/(double)iterations);
	return 0.0;
}

#endif
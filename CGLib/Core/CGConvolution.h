//author 2015 Wang Xinbo
//
//图像X,Y轴卷积，在HOG中提取图像梯度，与二维卷积核相比，速度更快

#pragma once

#include "CGImage.h"

namespace CG
{
	namespace Core
	{
		void CGComputeGradient(CGImage<float> *ImgDst, CGImage<float> *ImgIn);

		void CGComputeGradients_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn);

		void CGComputeGradNorm(CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, CGImage<float> *ImgIn);

		void CGComputeGradNorm_CPU(CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, CGImage<float> *ImgIn);

#ifndef COMPILE_WITHOUT_CUDA
		void CGComputeGradients_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn);

		void CGComputeGradNorm_CUDA(CGImage<float> *ImgGrad, CGImage<float> *ImgNorm, CGImage<float> *ImgIn);
#endif
	}
}
//author 2015 Wang Xinbo
//
//ͼ��X,Y��������HOG����ȡͼ���ݶȣ����ά�������ȣ��ٶȸ���

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
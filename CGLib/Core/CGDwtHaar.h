//author 2015 Wang Xinbo
//
//ͼ��X,Y��������HOG����ȡͼ���ݶȣ����ά�������ȣ��ٶȸ���

#pragma once

#include "CGImage.h"

namespace CG
{
	namespace Core
	{
		void CGDwtHaar(CGImage<float> *ImgDst, CGImage<float> *ImgIn);

		void CGDwtHaar_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn);

#ifndef COMPILE_WITHOUT_CUDA
		void CGDwtHaar_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn);
#endif
	}
}
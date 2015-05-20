//author 2015 Wang Xinbo
//

#pragma once

#include "CGImage.h"

namespace CG
{
	namespace Core
	{
		void CGDwtHaar(CGImage<float> *ImgDst, CGImage<float> *ImgIn, int haar_level = 1);

		void CGDwtHaar_CPU(CGImage<float> *ImgDst, CGImage<float> *ImgIn, int haar_level);

#ifndef COMPILE_WITHOUT_CUDA
		void CGDwtHaar_CUDA(CGImage<float> *ImgDst, CGImage<float> *ImgIn, int haar_level);
#endif
	}
}
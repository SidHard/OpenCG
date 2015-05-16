//author 2015 Wang Xinbo
//
//warpper function

#pragma once

#include "CGHogHistogram.h"
#include "CGHogSvmEvaluate.h"

using namespace CG::Core;

namespace CG
{
	namespace Hog
	{
		void CGHogExecute(std::vector<HogResult> &result, CGImage<float> *ImgIn);

		void CGHogSvmScore(CGImage<float> *svmScore, CGImage<float> *ImgIn, int scaleCount);

		void CGHogSvmScore_CPU(CGImage<float> *svmScore, CGImage<float> *ImgIn, int scaleCount);

#ifndef COMPILE_WITHOUT_CUDA
		__host__ void cgHogInit(float _svmBias, float* svmWeights, int svmWeightsCount);

		__host__ void CGHogSvmScore_CUDA(CGImage<float> *svmScore, CGImage<float> *ImgIn, int scaleCount);
#endif
	}
}
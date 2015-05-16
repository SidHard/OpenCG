//author 2015 Wang Xinbo
//
//svm_evaluate

#pragma once

#include "CGHogConfig.h"

using namespace CG::Core;

namespace CG
{
	namespace Hog
	{
		void CGHogSvmEvaluate(CGImage<float> *svmScore, CGImage<float> *hogHistogram, int imgWidth, int imgHight, int scaleId = 1);

		void CGHogSvmEvaluate_CPU(CGImage<float> *svmScore, CGImage<float> *hogHistogram, int imgWidth, int imgHight, int scaleId);

#ifndef COMPILE_WITHOUT_CUDA
		__host__ void cgInitSVM(float _svmBias, float* svmWeights, int svmWeightsCount);

		__host__ void CGHogSvmEvaluate_CUDA(CGImage<float> *svmScore, CGImage<float> *hogHistogram, int imgWidth, int imgHight, int scaleId);
#endif
	}
}
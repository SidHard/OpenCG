//author 2015 Wang Xinbo

#include "CGHogSvmEvaluate.h"

using namespace CG;
using namespace CG::Hog;

void Hog::CGHogSvmEvaluate_CPU(CGImage<float> *svmScore, CGImage<float> *hogHistogram, int imgWidth, int imgHight, int scaleId)
{}

void Hog::CGHogSvmEvaluate(CGImage<float> *svmScore, CGImage<float> *hogHistogram, int imgWidth, int imgHight, int scaleId)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGHogSvmEvaluate_CUDA(svmScore, hogHistogram, imgWidth, imgHight, scaleId);
#else
	CGHogSvmEvaluate_CPU(svmScore, hogHistogram, imgWidth, imgHight, scaleId);
#endif
}
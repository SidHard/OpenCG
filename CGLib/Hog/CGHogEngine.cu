//author 2015 Wang Xinbo

#include "CGHogEngine.h"

#ifndef COMPILE_WITHOUT_CUDA

using namespace CG;
using namespace CG::Core;
using namespace CG::Hog;

__host__ void Hog::cgHogInit(float _svmBias, float* svmWeights, int svmWeightsCount)
{
	Hog::cgInitHistogram();
	Hog::cgInitSVM(_svmBias, svmWeights, svmWeightsCount);
}

__host__ void 
Hog::CGHogSvmScore_CUDA(CGImage<float> *svmScore, CGImage<float> *ImgIn, int scaleCount)
{
	float currentScale = MIN_SCALE;

	CG::Core::CGImage<float> *ImgPyramid = new CG::Core::CGImage<float>(ImgIn->width, ImgIn->hight);
	CG::Core::CGImage<float> *ImgNorm = new CG::Core::CGImage<float>(ImgIn->width, ImgIn->hight);
	CG::Core::CGImage<float> *ImgGrad = new CG::Core::CGImage<float>(ImgIn->width, ImgIn->hight);
	//hogHistogram->ChangeSize((ImgIn->width / CELL_SIZE_W - 1) * BLOCK_SIZE_W * NO_BINS, (ImgIn->hight / CELL_SIZE_H - 1) * BLOCK_SIZE_H);
	CG::Core::CGImage<float> *hogHistogram = new CG::Core::CGImage<float>((ImgIn->width / CELL_SIZE_W - 1) * BLOCK_SIZE_W * NO_BINS, (ImgIn->hight / CELL_SIZE_H - 1) * BLOCK_SIZE_H);
	//svmScore->ChangeSize((ImgIn->width - WINDOW_SIZE_W) / CELL_SIZE_W + 1, scaleCount * ((ImgIn->hight - WINDOW_SIZE_H) / CELL_SIZE_H + 1));

	for (int i=0; i<scaleCount; i++)
	{
		CG::Core::CGPyramid(ImgPyramid, ImgIn, currentScale);

		CG::Core::CGComputeGradNorm(ImgGrad, ImgNorm, ImgPyramid);

		CG::Hog::CGHogHistogram(hogHistogram, ImgGrad, ImgNorm);

		CG::Hog::CGHogSvmEvaluate(svmScore, hogHistogram, ImgIn->width, ImgIn->hight, i);

		currentScale *= SCALE_STEP;
	}

	ImgPyramid->Free(); ImgNorm->Free(); ImgGrad->Free(); hogHistogram->Free();
}

#endif
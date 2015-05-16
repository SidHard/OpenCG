//author 2015 Wang Xinbo

#include "CGHogEngine.h"

using namespace CG;
using namespace CG::Hog;

void Hog::CGHogSvmScore_CPU(CGImage<float> *svmScore, CGImage<float> *ImgIn, int scaleCount)
{}

void Hog::CGHogSvmScore(CGImage<float> *svmScore, CGImage<float> *ImgIn, int scaleCount)
{
#ifndef COMPILE_WITHOUT_CUDA
	CGHogSvmScore_CUDA(svmScore, ImgIn, scaleCount);
#else
	CGHogSvmScore_CPU(svmScore, ImgIn, scaleCount);
#endif
}

void Hog::CGHogExecute(std::vector<HogResult> &hogResult, CGImage<float> *ImgIn)
{
	int scaleCount = (int)floor(logf(MAX_SCALE/MIN_SCALE)/logf(SCALE_STEP)) + 1;

	int hogWindowNumX = (ImgIn->width - WINDOW_SIZE_W)/CELL_SIZE_W + 1;
	int hogWindowNumY = (ImgIn->hight - WINDOW_SIZE_H)/CELL_SIZE_H + 1;

	CG::Core::CGImage<float> *svmScore = new CG::Core::CGImage<float>((ImgIn->width - WINDOW_SIZE_W) / CELL_SIZE_W + 1, scaleCount * ((ImgIn->hight - WINDOW_SIZE_H) / CELL_SIZE_H + 1));

	CGHogSvmScore(svmScore, ImgIn, scaleCount);

	svmScore->UpdateHostFromDevice();

	float currentScale = MIN_SCALE;

	for (int i=0; i<scaleCount; i++)
	{
		float *currentScaleOffset = svmScore->GetData(false) + i * hogWindowNumX * hogWindowNumY;

		for (int j=0; j<hogWindowNumX; j++)
		{
			for (int k=0; k<hogWindowNumY; k++)
			{
				float score = currentScaleOffset[k + j * hogWindowNumX];
				if (score > 0)
				{
					HogResult tempResult;

					tempResult.width = (int)floorf((float)WINDOW_SIZE_W * currentScale);
					tempResult.height = (int)floorf((float)WINDOW_SIZE_H * currentScale);

					tempResult.x = (int)ceilf(currentScale * k * CELL_SIZE_W);
					tempResult.y = (int)ceilf(currentScale * j * CELL_SIZE_H);

					tempResult.scale = currentScale;
					tempResult.score = score;

					hogResult.push_back(tempResult);
				}
			}
		}

		currentScale = currentScale * SCALE_STEP;
	}

	svmScore->Free();
}
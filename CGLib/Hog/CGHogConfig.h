//author 2015 Wang Xinbo
//hog

#pragma once

#include "..\Core\CGCore.h"
#include<vector>

#define BLOCK_SIZE_W 2
#define BLOCK_SIZE_H 2
#define CELL_SIZE_W 4
#define CELL_SIZE_H 4
#define WINDOW_SIZE_W 128
#define WINDOW_SIZE_H 64
#define NO_BINS 9

#define MIN_SCALE 1.0f
#define MAX_SCALE 2.0f
#define SCALE_STEP 1.05

struct HogResult
{
	float score;
	float scale;
	int width, height;
	int x, y;
};

inline int ClosestPowerOfTwo(int x) { x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x++; return x; }
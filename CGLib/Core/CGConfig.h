// author 2015 Wang Xinbo

#pragma once

//#define COMPILE_WITHOUT_CUDA

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef COMPILE_WITHOUT_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define UseGPU 1

#else

#define UseGPU 0

//struct float2
//{
//  float x, y;
//};

#endif

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef RADTODEG
#define RADTODEG 57.2957795
#endif

#ifndef INV_SQRT_2
#define INV_SQRT_2  0.70710678118654752440f;
#endif
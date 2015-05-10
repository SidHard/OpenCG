// author 2015 Wang Xinbo

#pragma once

#include "CGConfig.h"

namespace CG
{
	namespace Core
	{
		/*
		   CGImage
		*/
		template <typename T>
		class CGImage
		{
		private:
			bool allocateGPU;
			int isAllocated;

			/* CPU内存指针. */
			T* data_host;
			/* GPU指针. */
			T* data_device;
		public:
			/** 图像尺寸. */
			int width, hight;
			/** 像素总数. */
			int dataSize;

			/** 返回图像数据指针. */
			inline T* GetData(bool useGPU) { return useGPU ? data_device : data_host; }

			/** 返回图像数据指针. */
			inline const T* GetData(bool useGPU) const { return useGPU ? data_device : data_host; }

			/** 初始化一个空图像.*/
			explicit CGImage(bool allocateGPU = UseGPU)
			{
				this->isAllocated = false;
				this->noDims.x = this->noDims.y = 0;
				this->allocateGPU = allocateGPU;
			}

			/** 初始化一个给定尺寸的图像.*/
			CGImage(int width, int hight, bool allocateGPU = UseGPU)
			{
				this->isAllocated = false;
				this->allocateGPU = allocateGPU;
				Allocate(width, hight);
				this->Clear();
			}

			/** 为图像分配内存及显存.*/
			void Allocate(int width, int hight)
			{
				if (!this->isAllocated) {
					this->width = width;
					this->hight = hight;
					dataSize = width * hight;

					if (allocateGPU)
					{
#ifndef COMPILE_WITHOUT_CUDA
						cudaMallocHost((void**)&data_host, dataSize * sizeof(T));
						cudaMalloc((void**)&data_device, dataSize * sizeof(T));
#endif
					}
					else
					{ data_host = new T[dataSize]; }
				}

				isAllocated = true;
			}

			/** 为所有像素设置给定值. */
			void Clear(unsigned char  defaultValue = 0) 
			{ 
				memset(data_host, defaultValue, dataSize * sizeof(T)); 
#ifndef COMPILE_WITHOUT_CUDA
				if (allocateGPU) cudaMemset(data_device, defaultValue, dataSize * sizeof(T));
#endif
			}

			/** Resize.*/
			void ChangeSize(int new_width, int new_hight)
			{
				if ((width != new_width)||(hight != new_hight)||(!isAllocated)) {
					Free();
					Allocate(new_width, new_hight);
				}
			}

			/** CPU->GPU. */
			void UpdateDeviceFromHost() {
#ifndef COMPILE_WITHOUT_CUDA
				if (allocateGPU) cudaMemcpy(data_device, data_host, dataSize * sizeof(T), cudaMemcpyHostToDevice);
#endif
			}
			/** GPU->CPU. */
			void UpdateHostFromDevice() {
#ifndef COMPILE_WITHOUT_CUDA
				if (allocateGPU) cudaMemcpy(data_host, data_device, dataSize * sizeof(T), cudaMemcpyDeviceToHost);
#endif
			}

			/** 图像拷贝! */
			void SetFrom(const CGImage<T> *source, bool copyHost = true, bool copyDevice = false)
			{
				if (copyHost) memcpy(this->data_host, source->data_host, source->dataSize * sizeof(T));
#ifndef COMPILE_WITHOUT_CUDA
				if (copyDevice) cudaMemcpy(this->data_device, source->data_device, source->dataSize * sizeof(T), cudaMemcpyDeviceToDevice);
#endif
			}

			/** 释放内存及显存 */
			void Free()
			{
				if (this->isAllocated) {
					if (allocateGPU) {
#ifndef COMPILE_WITHOUT_CUDA
						cudaFree(data_device); 
						cudaFreeHost(data_host); 
#endif
					}
					else delete[] data_host;
				}

				this->isAllocated = false;
			}

			~CGImage() { this->Free(); }

			//
			CGImage(const CGImage&);
			CGImage& operator=(const CGImage&);
		};
	}
}

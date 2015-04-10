
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include<math_functions.h>
#include "readPPM.h"

#define WIDTH 10
#define HEIGHT 10
#define channels 3

#define BLOCK_SIZE 1024
#define HISTOGRAM_LENGTH 256

__device__ float clamp(float x, float start, float end)
{
	return min(max(x,start),end);
}

__device__ float color_correct(unsigned int val,float *cdf)
{
	return clamp(255*(cdf[val]-cdf[0])/(1-cdf[0]),0,255);
}

__global__ void convertToChar_kernel(float* input,unsigned char*output,int wd,int ht, int ch)
{
	int chIdx = 0;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*BLOCK_SIZE+ty;
	int col = blockIdx.x*BLOCK_SIZE+tx;
	int pixelIdx = (row*wd+col)*ch;
	
	for(chIdx = 0; chIdx < ch ; chIdx++)
	{
		if(row >= 0 && row < ht && col >= 0 && col < wd)
			output[pixelIdx+chIdx] = (unsigned char) (255 * input[pixelIdx+chIdx]);
		else
			output[pixelIdx+chIdx] = (unsigned char) 0;
		
		__syncthreads();
	}
	
}

__global__ void convertToFloat_kernel(unsigned char* input,float* output,int wd,int ht, int ch)
{
	int chIdx = 0;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*BLOCK_SIZE+ty;
	int col = blockIdx.x*BLOCK_SIZE+tx;
	int pixelIdx = (row*wd+col)*ch;
	
	for(chIdx = 0; chIdx < ch ; chIdx++)
	{
		if(row >= 0 && row < ht && col >= 0 && col < wd)
			output[pixelIdx+chIdx] = (float) (input[pixelIdx+chIdx]/255.0);
		else
			output[pixelIdx+chIdx] = (float) 0;
		
		__syncthreads();
	}
	
}

__global__ void convertRGB2GRAY_kernel(unsigned char* input, unsigned char* output, int wd, int ht, int ch)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*BLOCK_SIZE+ty;
	int col = blockIdx.x*BLOCK_SIZE+tx;
	int pixelIdx = (row*wd+col)*ch;
	
	
	
		if(row >= 0 && row < ht && col >= 0 && col < wd)
			output[row*wd+col] = (unsigned char) (0.21*input[pixelIdx]+0.71*input[pixelIdx+1]+0.07*input[pixelIdx+2]);
		else
			output[row*wd+col] = (unsigned char) 0;
		__syncthreads();
	
		
}

__global__ void histogram_kernel(unsigned char *buffer,long size,int wd, unsigned int *histogram)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*BLOCK_SIZE+ty;
	int col = blockIdx.x*BLOCK_SIZE+tx;
	
	int i = row*wd+col;
	int stride = blockDim.x*gridDim.x; 
	__shared__ unsigned int histogram_private[HISTOGRAM_LENGTH];
	
	if(threadIdx.x < HISTOGRAM_LENGTH)
		histogram_private[threadIdx.x] = 0;
	__syncthreads();
	
	while(i<size)
	{
		atomicAdd(&histogram_private[buffer[i]],1);
		i+=stride;
	}
	__syncthreads();
	if(threadIdx.x<HISTOGRAM_LENGTH)
		atomicAdd(&(histogram[threadIdx.x]),histogram_private[threadIdx.x]);
	
}

__global__ void cdf_kernel(unsigned int * histogram, float * cdf,int wd,int ht) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	__shared__ float cdf_private[HISTOGRAM_LENGTH];
	unsigned int t = threadIdx.x;
	unsigned int start = blockIdx.x*blockDim.x;

	if(t+start < HISTOGRAM_LENGTH)
		cdf_private[t+start] = (float) (histogram[t+start]/(wd*ht));
	
	__syncthreads();
	 
	
	for(unsigned int stride =1;stride<= HISTOGRAM_LENGTH; stride*=2)
	{
		int index = (t+1)*stride*2-1;
		if(index<HISTOGRAM_LENGTH)
			cdf_private[index]+=cdf_private[index-stride];
		__syncthreads();
	}
	
	for(unsigned int stride = HISTOGRAM_LENGTH/2;stride>0;stride/=2)
	{
		__syncthreads();
		int index = (t+1)*stride*2-1;
		if(index+stride < HISTOGRAM_LENGTH )
			cdf_private[index+stride]+=cdf_private[index];
	}
	
	__syncthreads();
	if(t+start < HISTOGRAM_LENGTH)
		cdf[t+start] = cdf_private[t+start];
}

__global__ void colorCorrection_kernel(unsigned char *input,unsigned char *output,float *cdf, int wd,int ht,int ch)
{
	int chIdx = 0;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*BLOCK_SIZE+ty;
	int col = blockIdx.x*BLOCK_SIZE+tx;
	int pixelIdx = (row*wd+col)*ch;
	
	for(chIdx = 0; chIdx < ch ; chIdx++)
	{
		if(row >= 0 && row < ht && col >= 0 && col < wd)
			output[pixelIdx+chIdx] = (unsigned char) color_correct(input[pixelIdx+chIdx],cdf);
		else
			output[pixelIdx+chIdx] = (unsigned char) 0;
		
		__syncthreads();
	}
}
	

void dispRes(float *arr)
{
	int i=0,j=0,k=0;
	printf("Results of the calculation\n");
	for(k=0;k<channels;k++){
		for(i=0;i<HEIGHT;i++){
			for(j=0;j<WIDTH;j++){
				printf("%2.1f ",arr[(i*WIDTH+j)*channels+k]);
			}
			
			printf("\n");
		}
		printf("k = %d\n",k);system("pause");
	}
}
int main(void)
{
	PPMImage *hostInputImageData;
	int imageWidth;
    int imageHeight;
    int imageChannels;
    //float * hostInputImageData;
    float * hostOutputImageData;
	float * deviceOutputImageData;
	float * deviceCDF;
	float * deviceInputImageData;
	unsigned char *  deviceInputImageCharData;
	unsigned char * deviceInputImageGrayData;
	unsigned char * deviceInputImageCCData;//colorcorrected image    
	unsigned int *deviceHistogram;
    

	
	//allocate Memory on the host
	 
	
	
	//load data to host memory
	hostInputImageData = readPPM("input.ppm");
	//loadData(hostInputImageData,hostOutputImageData,hostMaskData);

	//cuda memory allocation on the device
	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceInputImageCharData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceInputImageGrayData, imageWidth * imageHeight * sizeof(unsigned char));    
	cudaMalloc((void **) &deviceInputImageCCData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));    
	cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int)); 
	cudaMalloc((void **) &deviceCDF      , HISTOGRAM_LENGTH * sizeof(float));


	//cuda memory copy from host to device
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

	//grid and block dimensions
	dim3 DimGrid((imageWidth-1)/BLOCK_SIZE+1,(imageHeight-1)/BLOCK_SIZE+1,1);
	dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 DimBlockCDF(BLOCK_SIZE,1,1);

	//Launch kernels
	convertToChar_kernel<<<DimGrid,DimBlock>>>(deviceInputImageData,deviceInputImageCharData,imageWidth,imageHeight,imageChannels);
	convertRGB2GRAY_kernel<<<DimGrid,DimBlock>>>(deviceInputImageCharData,deviceInputImageGrayData,imageWidth,imageHeight,imageChannels);
	histogram_kernel<<<DimGrid,DimBlock>>>(deviceInputImageGrayData,imageWidth * imageHeight,imageWidth,deviceHistogram);
	cdf_kernel<<<1,DimBlockCDF>>>(deviceHistogram, deviceCDF,imageWidth,imageHeight);
	colorCorrection_kernel<<<DimGrid,DimBlock>>>(deviceInputImageCharData,deviceInputImageCCData,deviceCDF,imageWidth,imageHeight,imageChannels);
	convertToFloat_kernel<<<DimGrid,DimBlock>>>(deviceInputImageCCData,deviceOutputImageData,imageWidth,imageHeight,imageChannels);

	//cuda memory copy from device to host
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,imageWidth * imageHeight * imageChannels * sizeof(float),cudaMemcpyDeviceToHost);

	//dispRes(hostOutputImageDataCPU);
	dispRes(hostOutputImageData);

	free(hostInputImageData);
	free(hostOutputImageData);

	cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
	cudaFree(deviceInputImageCharData);
	cudaFree(deviceInputImageGrayData);
	cudaFree(deviceInputImageCCData);
	cudaFree(deviceHistogram);
	return 0;

}
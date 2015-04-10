#include <stdio.h>
#include<stdlib.h>
#include<string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define BLOCK_SIZE 1024
#define SECTION_SIZE 2*BLOCK_SIZE

__global__ void 
	listScanKernel(float * input, float * output, int len)
{
	__shared__ float list[SECTION_SIZE];
	unsigned int t = threadIdx.x;
	unsigned int start = blockIdx.x*blockDim.x;

	list[t] = ((t+start) < len ) ? input[t+start]:0.0f;
	list[t+blockDim.x] = ((start+t+blockDim.x) < len) ? input[start+t+blockDim.x]:0.0f;
 	
	for(unsigned int stride =1;stride<= BLOCK_SIZE; stride*=2)
	{
		int index = (t+1)*stride*2-1;
		if(index<SECTION_SIZE)
			list[index]+=list[index-stride];
		__syncthreads();
	}
	
	for(unsigned int stride = BLOCK_SIZE/2;stride>0;stride/=2)
	{
		__syncthreads();
		int index = (t+1)*stride*2-1;
		if(index+stride < SECTION_SIZE )
			list[index+stride]+=list[index];
	}
	
	__syncthreads();
	if(t+start < len)
		output[t+start] = list[t];
	
}
__global__ void
	loadSumArrayKernel(float *input, float *sumArray, int len)
{
	unsigned int t = threadIdx.x;
	unsigned int start = blockIdx.x*blockDim.x;
	unsigned int lastBlockId = (len-1)/BLOCK_SIZE;
	unsigned int lastThreadIdx = (len%BLOCK_SIZE-1);
	if(t+start<len)
	{
		if(blockIdx.x == lastBlockId)
			sumArray[blockIdx.x] = input[lastThreadIdx+start];
		else
		sumArray[blockIdx.x] = input[start+blockDim.x-1];
	}
}
__global__ void
	listScanSumKernel(float *input, float *output,int len)
{
	__shared__ float sumArray[SECTION_SIZE];
	unsigned int t = threadIdx.x;
	unsigned int start = blockIdx.x*blockDim.x;
	
	if(t+start<len && blockIdx.x>0)
	{
		
		output[t+start]+=input[blockIdx.x-1];
		__syncthreads();
	}
	
}

void totalCPU(float * input, float * output, int len)
{
	int i=0;
	output[0]=input[0];
	for(i=1;i<len;i++)
		output[i] = output[i-1]+input[i];

	printf("\n*****CPU calculation******\n");

}
void loadValue(char *FileInput,int len,float *a,float *b)
{
	FILE *file;
	int i=0;
	char buff[100];
	memset(b,0,len);

	file = fopen(FileInput,"r");
	if(!file)
	{
		printf("\nNo file found!");
		system("pause");
		exit(0);
	}

	while(fgets(buff,len,file))
	{
		a[i] = atof(buff);
		i++;
	}

	fclose(file);

}
void storeResult(char *fileOutput,float *arr,unsigned int len)
{
	FILE *file;
	int count=0;

	file = fopen(fileOutput,"w");
	if(!file)
	{
		printf("\nCannot create file!");
		system("pause");
		exit(0);
	}
	fprintf(file,"%d\n",len);
	for(count =0 ;count<len;count++)
	{
		fprintf(file,"%.0f\n",arr[count]);
	}
	fclose(file);
}
void dispRes(float *arr,int len)
{
	int i=0;
	printf("result = ");
	for(i=0;i<len;i++)
		printf("%4.0f ",arr[i]);	
	system("pause");
}
int main(int argc,char*argv[])
{
	float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
	float *deviceSumArray;
	float *deviceSumArrayOutput;
    float * deviceOutput;
    int numElements = (int) (atoi)(argv[3]); // number of elements in the input list

	hostInput = (float*)malloc(numElements*sizeof(float));
	hostOutput = (float*)malloc(numElements*sizeof(float));	

	//cuda memory allocation on the device
	cudaMalloc((void**)&deviceInput,numElements*sizeof(float));
	cudaMalloc((void**)&deviceOutput,numElements*sizeof(float));
	cudaMalloc((void**)&deviceSumArray,numElements*sizeof(float));
	cudaMalloc((void**)&deviceSumArrayOutput,numElements*sizeof(float));

	printf("Loading values to the array...\n");
	loadValue(argv[1],numElements,hostInput,hostOutput);

	//cuda memory copy from host to device
	cudaMemcpy(deviceInput,hostInput,numElements*sizeof(float),cudaMemcpyHostToDevice);

	//CPU equivalent
	totalCPU(hostInput,hostOutput,numElements);
	dispRes(hostOutput,numElements);

	printf("Calling CUDA kernel...\n");
	dim3 DimGrid((numElements-1)/BLOCK_SIZE+1,1,1);
	dim3 DimBlock(BLOCK_SIZE,1,1);	
	listScanKernel<<<DimGrid,DimBlock>>>(deviceInput,deviceOutput,numElements);
	loadSumArrayKernel<<<DimGrid,DimBlock>>>(deviceOutput,deviceSumArray,numElements);
	listScanKernel<<<DimGrid,DimBlock>>>(deviceSumArray,deviceSumArrayOutput,numElements);
	listScanSumKernel<<<DimGrid,DimBlock>>>(deviceSumArrayOutput,deviceOutput,numElements);

	//cuda memory copy from device to host
	cudaMemcpy(hostOutput,deviceOutput,numElements*sizeof(float),cudaMemcpyDeviceToHost);

	dispRes(hostOutput,numElements);

	storeResult(argv[2],hostOutput,numElements);

	free(hostInput);
	free(hostOutput);


	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	return 0;

}

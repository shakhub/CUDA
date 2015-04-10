#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void 
	vecAdd(float *a,float *b, float *c, int len)
{
	
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	
    if(i<len)	
		c[i] = a[i] + b[i];
	
}

void vecAdd_CPU(float *a,float *b, float *c, float len)
{
	int i=0;
	for(i=0;i<len;i++)
		c[i] =a[i]+b[i];

}

void loadValue(char *FileInput,int len,float *input)
{
	
	FILE *file;
	int i=0;
	char buff[100];
	
	file = fopen(FileInput,"r");
	
	if(!file)
	{
		printf("\nNo file found!");
		system("pause");
		exit(0);
	}
	
	while(fgets(buff,len,file))
	{
		input[i] = (float)atof(buff);
		i++;
	}
	
	fclose(file);
	
}
void storeResult(char *fileOutput,float *arr,unsigned int len)
{
	FILE *file;
	unsigned int count=0;

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
		fprintf(file,"%1.1f\n",arr[count]);
	}
	fclose(file);
}
int main(int argc, char* argv[])
{
	/*
	arg 1 inputFile 1
	arg 2 inputFile 2
	arg 3 outputFile
	arg 4 vector length
	*/
	
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float *d_A0,*d_B0,*d_C0;//device memory for stream0
	float *d_A1,*d_B1,*d_C1;//device memory for stream1
	int inputLength = (int) (atoi)(argv[4]); // number of elements in the input list
	int SegSize = inputLength/2;
	cudaStream_t stream0,stream1;
	
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);


	hostInput1 = (float*)malloc(inputLength*sizeof(float));
	hostInput2 = (float*)malloc(inputLength*sizeof(float));
	hostOutput = (float*)malloc(inputLength*sizeof(float));

	//cuda memory allocation on the device
	cudaMalloc((void**)&d_A0,SegSize*sizeof(float));
	cudaMalloc((void**)&d_B0,SegSize*sizeof(float));
	cudaMalloc((void**)&d_C0,SegSize*sizeof(float));
	
	cudaMalloc((void**)&d_A1,SegSize*sizeof(float));
	cudaMalloc((void**)&d_B1,SegSize*sizeof(float));
	cudaMalloc((void**)&d_C1,SegSize*sizeof(float));

	printf("Loading values to the array...\n");
	loadValue(argv[1],inputLength,hostInput1);
	loadValue(argv[2],inputLength,hostInput2);
	
	for(int i=0;i<inputLength;i+=SegSize*2)
	{
		cudaMemcpyAsync(d_A0, hostInput1+i, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
		
		cudaMemcpyAsync(d_B0, hostInput2+i, SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_A1, hostInput1+i+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1);
		cudaMemcpyAsync(d_B1, hostInput2+i+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1);
		vecAdd<<<(SegSize-1)/256+1, 256, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
		vecAdd<<<(SegSize-1)/256+1, 256, 0, stream1>>>(d_A1, d_B1,d_C1,SegSize);
		cudaMemcpyAsync(hostOutput+i, d_C0, SegSize*sizeof(float),cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(hostOutput+i+SegSize, d_C1, SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream1);
	}
	
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	cudaDeviceSynchronize();

	storeResult(argv[3],hostOutput,inputLength);

	/*
	free(hostInput1);
    free(hostInput2);
    free(hostOutput);
	*/
	cudaFree(d_A0);
	cudaFree(d_B0);
	cudaFree(d_C0);
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);
	return 0;

}
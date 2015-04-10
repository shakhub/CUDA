#include <stdio.h>
#include "cuda_profiler_api.h"

#define SIZE 20480

__global__ void 
	vecAdd(int *a,int *b, int *c, int len)
{
	
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	
    if(i<len)	c[i] = a[i] + b[i];
	
}

void vecAdd_CPU(int *a,int *b, int *c, int len)
{
	int i=0;
	for(i=0;i<len;i++)
		c[i] =a[i]+b[i];

}
void loadVal(int *a, int *b, int *c, int len)
{
	int i=0;
	for(i=0;i<len;i++)
	{
		a[i] = i*5;
		b[i] = i*6;
		c[i] = 0;
	}
}

void dispRes(int *arr)
{
	int i=0;
	printf("Results of the first 10 elements\n");
	for(i=0;i<10;i++)
	{
		printf("%d, ",arr[i]);
	}
}
int main(void)
{
	int *a,*b,*c;
	int *d_a,*d_b,*d_c; //device variables 
	
	a = (int*)malloc(SIZE*sizeof(int));
	b = (int*)malloc(SIZE*sizeof(int));
	c = (int*)malloc(SIZE*sizeof(int));

	//cuda memory allocation on the device
	cudaMalloc((void**)&d_a,SIZE*sizeof(int));
	cudaMalloc((void**)&d_b,SIZE*sizeof(int));
	cudaMalloc((void**)&d_c,SIZE*sizeof(int));

	printf("Loading values to the array...\n");
	loadVal(a,b,c,SIZE);

	//cuda memory copy from host to device
	cudaMemcpy(d_a,a,SIZE*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,SIZE*sizeof(int),cudaMemcpyHostToDevice);

	printf("Calling vector adding function...\n");
	dim3 DimGrid((SIZE-1)/256+1,1,1);
	dim3 DimBlock(256,1,1);

	cudaProfilerStart();
	vecAdd<<<DimGrid,DimBlock>>>(d_a,d_b,d_c,SIZE);
	cudaProfilerStop();

	//CPU equivalent
	vecAdd_CPU(a,b,c,SIZE);
	
	//cuda memory copy from device to host
	cudaMemcpy(c,d_c,SIZE*sizeof(int),cudaMemcpyDeviceToHost);

	dispRes(c);
system("PAUSE");
	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;

}
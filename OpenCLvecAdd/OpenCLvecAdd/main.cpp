#include <stdio.h>
#include <stdlib.h>

#include"main.h"
//
// OpenCL source code
const char* OpenCLSource = 
"__kernel void VectorAdd(__global int* c, __global int* a,__global int* b) \n"
"{\n"
" // Index of the elements to add \n"
" unsigned int n = get_global_id(0);\n"
" // Sum the nth element of vectors a and b and store in c \n"
" c[n] = a[n] + b[n];\n"
"}\n"
;

void initOpenCL(void)
{
	char cBuffer[1024];

	error = clGetPlatformIDs(1, &cpPlatform, NULL);
	printf("clGetPlatformIDs %d\n",error);

	error = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	printf("clGetPlatformIDs %d\n",error);

	clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
	printf("CL_DEVICE_NAME: %s\n", cBuffer);
	clGetDeviceInfo(cdDevice, CL_DRIVER_VERSION, sizeof(cBuffer), &cBuffer, NULL);
	printf("CL_DRIVER_VERSION: %s\n\n", cBuffer);
	{
		// Create a context to run OpenCL enabled GPU
		cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) cpPlatform,0};
		GPUContext = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
		printf("clCreateContextFromType %d\n",error);
		// Create a command-queue on the GPU device
		cqCommandQueue = clCreateCommandQueue(GPUContext, cdDevice, 0, NULL);
	}
}
// Main function
// ************************************************************
int main(int argc, char **argv)
{
	 
     // Two integer source vectors in Host memory
     int HostVector1[SIZE], HostVector2[SIZE];
     //Output Vector
     int HostOutputVector[SIZE];
	 cl_program OpenCLProgram;
	 cl_kernel OpenCLVectorAdd;
	 cl_mem GPUVector1,GPUVector2,GPUOutputVector;
	 size_t WorkSize;

     // Initialize with some interesting repeating data
     for(int c = 0; c < SIZE; c++)
     {
          HostVector1[c] = InitialData1[c%20];
          HostVector2[c] = InitialData2[c%20];
          HostOutputVector[c] = 0;
     }
	
	 initOpenCL();
	 // Allocate GPU memory for source vectors AND initialize from CPU memory
	 GPUVector1 = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, HostVector1, NULL);
	 GPUVector2 = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, HostVector2, NULL);
	 // Allocate output memory on GPU
	 GPUOutputVector = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE, NULL, NULL);

	 // Create OpenCL program with source code
     OpenCLProgram = clCreateProgramWithSource(GPUContext, 1, &OpenCLSource, NULL, NULL);
     // Build the program (OpenCL JIT compilation)
     error = clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);
	 printf("clBuildProgram %d\n",error);
     // Create a handle to the compiled OpenCL function (Kernel)
     OpenCLVectorAdd = clCreateKernel(OpenCLProgram, "VectorAdd", NULL);
     
	 // In the next step we associate the GPU memory with the Kernel arguments
     clSetKernelArg(OpenCLVectorAdd, 0, sizeof(cl_mem), (void*)&GPUOutputVector);
     clSetKernelArg(OpenCLVectorAdd, 1, sizeof(cl_mem), (void*)&GPUVector1);
     clSetKernelArg(OpenCLVectorAdd, 2, sizeof(cl_mem), (void*)&GPUVector2);
     
	 // Launch the Kernel on the GPU
     // This kernel only uses global data
     WorkSize = SIZE; // one dimensional Range
     clEnqueueNDRangeKernel(cqCommandQueue, OpenCLVectorAdd, 1, NULL,&WorkSize, NULL, 0, NULL, NULL);
     // Copy the output in GPU memory back to CPU memory
     clEnqueueReadBuffer(cqCommandQueue, GPUOutputVector, CL_TRUE, 0,SIZE * sizeof(int), HostOutputVector, 0, NULL, NULL);
     
	 // Cleanup
     clReleaseKernel(OpenCLVectorAdd);
     clReleaseProgram(OpenCLProgram);
     clReleaseCommandQueue(cqCommandQueue);
     clReleaseContext(GPUContext);
     clReleaseMemObject(GPUVector1);
     clReleaseMemObject(GPUVector2);
     clReleaseMemObject(GPUOutputVector);
	 
     for( int i =0 ; i < SIZE; i++)
          printf("[%d + %d = %d]\n",HostVector1[i], HostVector2[i], HostOutputVector[i]);
	
	 system("PAUSE");
     return 0;
}

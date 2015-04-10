#include <CL/cl.h>

#define SIZE 100// Number of elements in the vectors to be added

// Some interesting data for the vectors
int InitialData1[20] = {37,50,54,50,56,0,43,43,74,71,32,36,16,43,56,100,50,25,15,17};
int InitialData2[20] = {35,51,54,58,55,32,36,69,27,39,35,40,16,44,55,14,58,75,18,15};

 cl_int error = CL_SUCCESS; 
 cl_platform_id cpPlatform;//Get an OpenCL platform
 cl_device_id cdDevice;// Get a GPU device
 cl_context GPUContext;
 cl_command_queue cqCommandQueue;

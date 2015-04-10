#include <stdio.h>
#include<stdlib.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#define BLOCK_SIZE 64

__global__ void 
	totalKernel(float * input, float * output, int len)
{
	__shared__ float partialSum[2*BLOCK_SIZE];
	unsigned int t = threadIdx.x;
	unsigned int start = blockIdx.x*blockDim.x*2;

	if((start+t)<len)
	{
		partialSum[t] = input[start+t];
		if(start+t+blockDim.x < len)
			partialSum[blockDim.x+t] = input[start+t+blockDim.x];
		else
			partialSum[blockDim.x+t] =0;
	}
	else
	{
		partialSum[t] = 0;
		partialSum[blockDim.x+t] = 0;

	}
	__syncthreads();
	
	for(unsigned int stride = blockDim.x; stride >0; stride/=2){
		__syncthreads();
		if(t<stride)
			partialSum[t]+=partialSum[t+stride];		
	}
	//__syncthreads();
	output[blockIdx.x] = partialSum[0];
}

void totalCPU(float * input, float * output, int len)
{
	int i=0;
	output[0]=0;
	for(i=0;i<len;i++)
		output[0] +=input[i];

}
void loadVal(float *a, float *b, int len)
{
	/*
	int i=0;
	char buff[256];
	FILE *file;

	file = fopen("dataSetRaw0.txt","r");

	if(!file)
	{
		printf("No file found");
		system("Pause");
		exit(0);
	}

    while(fgets(buff,len,file))
	{
		a[i] = atof(buff);
		i++;
	}
	
	for(i=0;i<len;i++)
	{
		a[i] = i;		
	}
	
	fclose(file);
	*/
	a[0]=7;
a[1]=9;
a[2]=1;
a[3]=4;
a[4]=2;
a[5]=8;
a[6]=10;
a[7]=5;
a[8]=10;
a[9]=7;
a[10]=5;
a[11]=7;
a[12]=8;
a[13]=6;
a[14]=4;
a[15]=6;
a[16]=6;
a[17]=3;
a[18]=4;
a[19]=0;
a[20]=1;
a[21]=10;
a[22]=5;
a[23]=8;
a[24]=7;
a[25]=0;
a[26]=2;
a[27]=9;
a[28]=2;
a[29]=8;
a[30]=4;
a[31]=3;
a[32]=2;
a[33]=1;
a[34]=4;
a[35]=10;
a[36]=3;
a[37]=9;
a[38]=6;
a[39]=9;
a[40]=4;
a[41]=7;
a[42]=3;
a[43]=3;
a[44]=3;
a[45]=1;
a[46]=5;
a[47]=5;
a[48]=0;
a[49]=7;
a[50]=7;
a[51]=2;
a[52]=5;
a[53]=7;
a[54]=9;
a[55]=5;
a[56]=8;
a[57]=5;
a[58]=0;
a[59]=10;
a[60]=3;
a[61]=9;
a[62]=5;
a[63]=10;
a[64]=8;
a[65]=4;
a[66]=8;
a[67]=8;
a[68]=2;
a[69]=6;
a[70]=9;
a[71]=6;
a[72]=9;
a[73]=0;
a[74]=9;
a[75]=7;
a[76]=3;
a[77]=1;
a[78]=8;
a[79]=7;
a[80]=0;
a[81]=10;
a[82]=9;
a[83]=8;
a[84]=7;
a[85]=10;
a[86]=9;
a[87]=1;
a[88]=4;
a[89]=3;
a[90]=1;
a[91]=8;
a[92]=2;
a[93]=4;
a[94]=8;
a[95]=1;
a[96]=1;
a[97]=2;
a[98]=7;
a[99]=10;
a[100]=6;
a[101]=10;
a[102]=0;
a[103]=0;
a[104]=2;
a[105]=2;
a[106]=5;
a[107]=1;
a[108]=6;
a[109]=10;
a[110]=2;
a[111]=2;
a[112]=8;
a[113]=10;
a[114]=10;
a[115]=9;
a[116]=4;
a[117]=5;
a[118]=9;
a[119]=3;
a[120]=3;
a[121]=4;
a[122]=6;
a[123]=8;
a[124]=8;
a[125]=9;
a[126]=9;
a[127]=3;
a[128]=1;
a[129]=4;
a[130]=10;
a[131]=7;
a[132]=7;
a[133]=0;
a[134]=4;
a[135]=4;
a[136]=7;
a[137]=7;
a[138]=0;
a[139]=1;
a[140]=5;
a[141]=4;
a[142]=4;
a[143]=8;
a[144]=9;
a[145]=10;
a[146]=10;
a[147]=10;
a[148]=3;
a[149]=4;
a[150]=10;
a[151]=6;
a[152]=9;
a[153]=7;
a[154]=10;
a[155]=10;
a[156]=2;
a[157]=8;
a[158]=5;
a[159]=5;
a[160]=7;
a[161]=9;
a[162]=1;
a[163]=3;
a[164]=6;
a[165]=6;
a[166]=5;
a[167]=3;
a[168]=9;
a[169]=6;
a[170]=6;
a[171]=7;
a[172]=1;
a[173]=4;
a[174]=8;
a[175]=8;
a[176]=6;
a[177]=2;
a[178]=9;
a[179]=8;
a[180]=5;
a[181]=5;
a[182]=5;
a[183]=3;
a[184]=0;
a[185]=8;
a[186]=0;
a[187]=4;
a[188]=8;
a[189]=7;
a[190]=9;
a[191]=10;
a[192]=0;
a[193]=5;
a[194]=10;
a[195]=8;
a[196]=3;
a[197]=1;
a[198]=8;
a[199]=3;
a[200]=1;
a[201]=10;
a[202]=5;
a[203]=8;
a[204]=2;
a[205]=6;
a[206]=1;
a[207]=7;
a[208]=10;
a[209]=7;
a[210]=9;
a[211]=5;
a[212]=9;
a[213]=3;
a[214]=1;
a[215]=5;
a[216]=0;
a[217]=9;
a[218]=3;
a[219]=6;
a[220]=5;
a[221]=10;
a[222]=7;
a[223]=5;
a[224]=10;
a[225]=8;
a[226]=7;
a[227]=3;
a[228]=1;
a[229]=9;
a[230]=5;
a[231]=8;
a[232]=7;
a[233]=8;
a[234]=5;
a[235]=3;
a[236]=3;
a[237]=0;
a[238]=3;
a[239]=1;
a[240]=10;
a[241]=6;
a[242]=8;
a[243]=8;
a[244]=7;
a[245]=7;
a[246]=1;
a[247]=5;
a[248]=10;
a[249]=1;
a[250]=8;
a[251]=10;
a[252]=1;
a[253]=9;
a[254]=1;
a[255]=1; 
}

void dispRes(float arr)
{
	printf("result = ");
	printf("%f ",arr);	
	system("pause");
}
int main(int argc,char*argv[])
{
	int ii;
	float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements = 256; // number of elements in the input list
    int numOutputElements = 0; // number of elements in the output list

		numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }

	hostInput = (float*)malloc(numInputElements*sizeof(float));
	hostOutput = (float*)malloc(numOutputElements*sizeof(float));
	

	//cuda memory allocation on the device
	cudaMalloc((void**)&deviceInput,numInputElements*sizeof(float));
	cudaMalloc((void**)&deviceOutput,numOutputElements*sizeof(float));
	

	printf("Loading values to the array...\n");
	loadVal(hostInput,hostOutput,numInputElements);

	//cuda memory copy from host to device
	cudaMemcpy(deviceInput,hostInput,numInputElements*sizeof(float),cudaMemcpyHostToDevice);

	//CPU equivalent
	//totalCPU(hostInput,hostOutput,numInputElements);
	//dispRes(hostOutput[0]);

	printf("Calling CUDA kernel...\n");
	dim3 DimGrid((numInputElements-1)/BLOCK_SIZE+1,1,1);
	dim3 DimBlock(BLOCK_SIZE,1,1);	
	totalKernel<<<DimGrid,DimBlock>>>(deviceInput,deviceOutput,numInputElements);	

	//cuda memory copy from device to host
	cudaMemcpy(hostOutput,deviceOutput,numOutputElements*sizeof(float),cudaMemcpyDeviceToHost);

	for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

	dispRes(hostOutput[0]);

	free(hostInput);
	free(hostOutput);


	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	return 0;

}
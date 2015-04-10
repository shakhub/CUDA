
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define WIDTH 10
#define HEIGHT 10
#define channels 3
#define Mask_width  5
#define Mask_radius Mask_width/2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH+Mask_width-1)
#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))
#define clamp(x) (min(max((x),0.0),1.0))

void imageConvolution(float *input,float *output,const float* __restrict__ M,int width, int height, int ch)
{
	int i=0,j=0,k=0,x=0,y=0,xOffset=0,yOffset=0;
   float accum =0.0,maskValue =0.0,imagePixel =0.0;
   
	for( i=0 ;i<height;i++){
      for( j=0;j< width;j++){
        for(k=0;k<ch;k++){
          accum = 0;
          for(y = 0 ;y< Mask_width;y++){
            for(x= 0;x< Mask_width; x++){
              xOffset = j + x - Mask_radius;
              yOffset = i + y - Mask_radius;
              if (xOffset>=0 && xOffset < width && yOffset>=0 && yOffset < height){
                imagePixel = input[(yOffset * width + xOffset) * channels + k];
                maskValue  = M[y*Mask_width+x];
                accum += imagePixel * maskValue;
              }
			  else
				  accum +=0;
            }			
          }
          output[(i * width + j)*channels + k] = accum;// (float) clamp(accum);
          
        }
      }
    }
}

__global__ void imageTiledConvolution_kernel(float *input,float *output,const float * __restrict__ M,int width, int height, int ch)
{
	int i=0,j=0;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*O_TILE_WIDTH+ty;
	int col_o = blockIdx.x*O_TILE_WIDTH+tx;
	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;
	float cValue = 0.0f;
	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH][channels]; 
	
	for(int chIdx=0;chIdx<ch;chIdx++){
		
		if(row_i>=0 && row_i<height && col_i>=0 && col_i<width){
				Ns[ty][tx][chIdx] = input[(row_i*width+col_i)*ch+chIdx];		
		}else{
				Ns[ty][tx][chIdx] = 0.0f;
		}
		__syncthreads();
		
		cValue = 0.0f;
		if(ty<O_TILE_WIDTH && tx<O_TILE_WIDTH){
			for( i=0;i<Mask_width;i++){
				for( j=0;j<Mask_width;j++){
					cValue +=M[i*Mask_width+j]*Ns[ty+i][tx+j][chIdx];
				}
			}
		}			
		
		__syncthreads();
		
		if(row_o<height && col_o<width && ty<O_TILE_WIDTH && tx<O_TILE_WIDTH)
			output[(row_o*width+col_o)*ch+chIdx] = cValue;//min(max(cValue,0),1);
		
	}	
	
}


void loadData(float *input,float *output,float *maskData)
{
  int i=0; 
  for(i=0;i<WIDTH*HEIGHT*channels;i++)
    input[i] = 1.0;
  
  for(i=0;i<WIDTH*HEIGHT*channels;i++)
    output[i] = 0.0;


  for(i=0;i<Mask_width *Mask_width ;i++)
    maskData[i] = 1.0;


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
	int maskRows = Mask_width;
	int maskColumns = Mask_width;
	int imageChannels = channels;
	int imageWidth = WIDTH;
	int imageHeight = HEIGHT;
    float * hostInputImageData;
    float * hostOutputImageData;
	float * hostOutputImageDataCPU;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

	
	//allocate Memory on the host
	hostInputImageData = (float*)malloc(imageWidth*imageHeight*channels*sizeof(float));
	hostOutputImageData = (float*)malloc(imageWidth*imageHeight*channels*sizeof(float));
	hostOutputImageDataCPU = (float*)malloc(imageWidth*imageHeight*channels*sizeof(float));
	hostMaskData = (float*)malloc(Mask_width*Mask_width*sizeof(float));

	//load data to host memory
	loadData(hostInputImageData,hostOutputImageData,hostMaskData);


	//cuda memory allocation on the device
	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));


	//cuda memory copy from host to device
	cudaMemcpy(deviceInputImageData,hostInputImageData,imageWidth * imageHeight * imageChannels * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,hostMaskData,maskRows * maskColumns * sizeof(float),cudaMemcpyHostToDevice);

	
	dim3 DimGrid((imageWidth-1)/O_TILE_WIDTH+1,(imageHeight-1)/O_TILE_WIDTH+1,1);
	dim3 DimBlock(BLOCK_WIDTH,	BLOCK_WIDTH,1);	

	imageTiledConvolution_kernel<<<DimGrid,DimBlock>>>(deviceInputImageData,deviceOutputImageData,deviceMaskData,imageWidth,imageHeight,imageChannels);

	imageConvolution(hostInputImageData,hostOutputImageDataCPU,hostMaskData,imageWidth,imageHeight,imageChannels);

	//cuda memory copy from device to host
	cudaMemcpy(hostOutputImageData,deviceOutputImageData,imageWidth * imageHeight * imageChannels * sizeof(float),cudaMemcpyDeviceToHost);

	//dispRes(hostOutputImageDataCPU);
	dispRes(hostOutputImageData);

	free(hostInputImageData);
	free(hostOutputImageData);
	free(hostOutputImageDataCPU);
	free(hostMaskData);

	cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);
	return 0;

}
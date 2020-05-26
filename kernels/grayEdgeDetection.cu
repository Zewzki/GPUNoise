#include "math.h"

__constant__ int sobelV[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__constant__ int sobelH[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

extern "C"
__global__ void grayEdgeDetection(int * output, int width, int thresh) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int pixelIndex = idx + (idy * width);
	
	int pixels[9];
	
	for(int i = -1; i <= 1; i++) {
		for(int j = -1; j <= 1; j++) {
			pixels[((i + 1) * 3) + (j + 1)] = output[(idx + j) + ((idy + i) * width)];
		}
	}
	
	/*
	pixels[0] = output[(idx - 1) + ((idy - 1) * width)];
	pixels[1] = output[(idx) + ((idy - 1) * width)];
	pixels[2] = output[(idx + 1) + ((idy - 1) * width)];
	
	pixels[3] = output[(idx -  1) + (idy * width)];
	pixels[4] = output[(idx) + (idy * width)];
	pixels[5] = output[(idx +  1) + (idy * width)];
	
	pixels[6] = output[(idx - 1) + ((idy + 1) * width)];
	pixels[7] = output[(idx) + ((idy + 1) * width)];
	pixels[8] = output[(idx + 1) + ((idy + 1) * width)];
	*/
	
	int vertSum = 0;
	int horzSum = 0;
	
	for(int i = 0; i < 9; i++) {
		int grayVal = pixels[i] & 0xFF;
		vertSum = vertSum + (grayVal * sobelV[i]);
		horzSum = horzSum + (grayVal * sobelH[i]);
	}
	
	int sum = (int) (sqrt((double)((vertSum * vertSum) + (horzSum * horzSum))));
	if(sum > 255) sum = 255;
	if(sum < thresh) sum = 0;
	
	output[pixelIndex] = sum;
	
	
}
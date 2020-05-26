#define width 1024
#define height 1024

#include "math.h"

__constant__ int perm[] = {151,160,137,91,90,15,
 		   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
 		   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
 		   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
 		   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
 		   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
 		   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
 		   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
 		   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
 		   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
 		   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
 		   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
 		   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180};

__constant__ int grad3[] = { 1, 1, 0, -1, 1, 0, 1, -1,
							0, -1, -1, 0, 1, 0, 1, -1,
							0, 1, 1, 0, -1, -1, 0, -1,
							0, 1, 1, 0, -1, 1, 0, 1, -1,
							0, -1, -1 };

__device__ double lerp(double y1, double y2, double mu) {
	return y2 + y1 * (mu - y2);
}
    
__device__ double fade(double x) {
	return x * x * x * (x * (x * 6 - 15) + 10);
}

__device__ int fastFloor(double x) {
	return x > 0 ? (int) x : (int) x - 1;
}

__device__ double grad(int hash, double x, double y) {
	int h = hash & 15;
	double u = h < 8 ? x : y;
	return ((h & 1) == 0 ? u : -u);
}
__device__ double grad(int hash, double x, double y, double z) {
	int h = hash & 15; // CONVERT LO 4 BITS OF HASH CODE
	double u = h < 8 ? x : y, // INTO 12 GRADIENT DIRECTIONS.
			v = h < 4 ? y : h == 12 || h == 14 ? x : z;
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__device__ double map(double input, int min1, int max1, int min2, int max2) {
	return (((input - min1) / (max1 - min1)) * (max2 - min2)) + min2;
}

extern "C"
__global__ void random(int * output, double x, double y) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int pixelIndex = (idy * blockDim.y) + idx;
	int pixelValue = ((int)(idy + y) * blockDim.x) + (int)(idx + x);
	
	output[pixelIndex] = perm[pixelValue % 256];

}

extern "C"
__global__ void noise3D(int * output, double x, double y, double z) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int pixelIndex = (idy * width) + idx;
	
	x += idx;
	y += idx;
	
	int X = (int) floor(x) & 255;
	int Y = (int) floor(y) & 255;
	int Z = (int) floor(z) & 255;
	
	x -= floor(x);
	y -= floor(y);
	z -= floor(z);
	
	double u = fade(x);
	double v = fade(y);
	double w = fade(z);
	
	int A = perm[X] + Y;
	int AA = perm[A] + Z;
	int AB = perm[A + 1] + Z;
	int B = perm[X + 1] + Y;
	int BA = perm[B] + Z;
	int BB = perm[B + 1] + Z;
	
	double grad1 = grad(perm[BA], x - 1, y, z);
	double grad2 = grad(perm[BB], x - 1, y, z - 1);
	double grad3 = grad(perm[BA + 1], x - 1, y, z - 1);
	double grad4 = grad(perm[BB + 1], x - 1, y - 1, z - 1);
	
	double smallLerp1 = lerp(u, grad(perm[AA], x, y, z), grad1);
	double smallLerp2 = lerp(u, grad(perm[AB], x, y - 1, z), grad2);
	double smallLerp3 = lerp(u, grad(perm[AA + 1], x, y, z - 1), grad3);
	double smallLerp4 = lerp(u, grad(perm[AB + 1], x, y - 1, z - 1), grad4);
	
	double medLerp1 = lerp(v, smallLerp1, smallLerp2);
	double medLerp2 = lerp(v, smallLerp3, smallLerp4);
	
	double bigLerp = lerp(w, medLerp1, medLerp2);
	
	
	double value = lerp(w,
					lerp(v,
						lerp(u, grad(perm[AA], x, y, z),
							grad(perm[BA], x - 1, y, z)),
						lerp(u, grad(perm[AB], x, y - 1, z),
							grad(perm[BB], x - 1, y, z - 1))),
					lerp(v,
						lerp(u, grad(perm[AA + 1], x, y, z - 1),
							grad(perm[BA + 1], x - 1, y, z - 1)),
						lerp(u, grad(perm[AB + 1], x, y - 1, z - 1),
							grad(perm[BB + 1], x - 1, y - 1, z - 1))));					
	
	output[pixelIndex] = map(value, -1, 1, 0, 255);
	
}

extern "C"
__global__ void noise2D(int * output, double x, double y) {

	int idx = threadIdx.x;
	int idy = blockIdx.x;
	
	int pixelIndex = idx + (blockIdx.x * blockDim.x);
	
	x += idx;
	y += idy;
	
	int xi = (fastFloor(x) & 255);
    int yi = (fastFloor(y) & 255);
    int g1 = perm[perm[xi] + yi];
    int g2 = perm[perm[xi + 1] + yi];
    int g3 = perm[perm[xi] + yi + 1];
    int g4 = perm[perm[xi + 1] + yi + 1];
    
    double xf = x - fastFloor(x);
    double yf = y - fastFloor(y);
    
    double d1 = grad(g1, xf, yf);
    double d2 = grad(g2, xf - 1, yf);
    double d3 = grad(g3, xf, yf - 1);
    double d4 = grad(g4, xf - 1, yf - 1);
    	
    double u = fade(xf);
    double v = fade(yf);
    
    double x1Inter = lerp(u, d1, d2);
    double x2Inter = lerp(u, d3, d4);
    double yInter = lerp(v, x1Inter, x2Inter);
    
    output[pixelIndex] = map(d1, 0, 255, 0, 255);
    
}

extern "C"
__global__ void woodNoise(int * output, double x, double y) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	x += (id % blockDim.x);
	y += (int) (id / blockDim.x);

	int xRanged = fastFloor(x) & 255;
	int yRanged = fastFloor(y) & 255;

	double xPos = x - fastFloor(x);
	double yPos = y - fastFloor(y);

	double xFade = fade(xPos);
	double yFade = fade(yPos);

	// HASH COORDINATES
	int A = perm[xRanged] + yRanged;
	int B = perm[xRanged + 1] + yRanged;
	int C = perm[yRanged] + xRanged;
	int D = perm[yRanged + 1] + xRanged;

	double gradA = grad(perm[A], x, y);
	double gradB = grad(perm[B], x, y);
	double gradC = grad(perm[C], x, y);
	double gradD = grad(perm[D], x, y);

	double lerp1 = lerp(xFade, gradA, gradB);
	double lerp2 = lerp(xFade, gradC, gradD);

	double finalLerp = lerp(yFade, lerp1, lerp2);
	
	finalLerp = fabs(finalLerp);
	finalLerp = fmod(finalLerp, 2.0);
	//finalLerp -= 1;

	// finalLerp %= 1;

	// System.out.println(finalLerp);

	output[id] = map(finalLerp, 0, 1, 0, 255);

}
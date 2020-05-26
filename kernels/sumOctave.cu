#define iterations 8
#define persistance .6
#define scale .001
#define high 255
#define low 0
#define PI 3.1415926535
#define noiseType 0

#include "math.h"

__constant__ int perm[] = {151,160,137,91,90,15,
 		   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
 		   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
 		   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
 		   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
 		   102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
 		   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
 		   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
 		   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
 		   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
 		   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
 		   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
 		   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180};

__constant__ int grad3[] = {1, 1, 0, -1, 1, 0, 1, -1,
							0, -1, -1, 0, 1, 0, 1, -1,
							0, 1, 1, 0, -1, -1, 0, -1,
							0, 1, 1, 0, -1, 1, 0, 1, -1,
							0, -1, -1 };

__device__ double mix(double a, double b, double t) {
	return (1 - t) * a + t * b;
}

__device__ double fade(double t) {
	return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ double dot(int * g, double x, double y) {
	return g[0] * x + g[1] * y;
}

__device__ double dot(int * g, double x, double y, double z) {
	return g[0] * x + g[1] * y + g[2] * z;
}

__device__ int fastFloor(double x) {
	return x > 0 ? (int) x : (int) x - 1;
}

__device__ double map(double input, int min1, int max1, int min2, int max2) {
	return (((input - min1) / (max1 - min1)) * (max2 - min2)) + min2;
}

__device__ double noise3D(double x, double y, double z) {
	
    int X = fastFloor(x);
    int Y = fastFloor(y);
    int Z = fastFloor(z);
    
    //int X = floor(x);
    //int Y = floor(y);
    //int Z = floor(z);
    
    x = x - X;
    y = y - Y;
    z = z - Z;
    
    X = X & 255;
    Y = Y & 255;
    Z = Z & 255;
    
    int gi000 = perm[X + perm[Y + perm[Z]]] % 12;
	int gi001 = perm[X + perm[Y + perm[Z + 1]]] % 12;
	int gi010 = perm[X + perm[Y + 1 + perm[Z]]] % 12;
	int gi011 = perm[X + perm[Y + 1 + perm[Z + 1]]] % 12;
	int gi100 = perm[X + 1 + perm[Y + perm[Z]]] % 12;
	int gi101 = perm[X + 1 + perm[Y + perm[Z + 1]]] % 12;
	int gi110 = perm[X + 1 + perm[Y + 1 + perm[Z]]] % 12;
	int gi111 = perm[X + 1 + perm[Y + 1 + perm[Z + 1]]] % 12;
	
	double n000 = dot(grad3 + gi000, x, y, z);
	double n100 = dot(grad3 + gi100, x - 1, y, z);
	double n010 = dot(grad3 + gi010, x, y - 1, z);
	double n110 = dot(grad3 + gi110, x - 1, y - 1, z);
	double n001 = dot(grad3 + gi001, x, y, z - 1);
	double n101 = dot(grad3 + gi101, x - 1, y, z - 1);
	double n011 = dot(grad3 + gi011, x, y - 1, z - 1);
	double n111 = dot(grad3 + gi111, x - 1, y - 1, z - 1);
	
	double u = fade(x);
	double v = fade(y);
	double w = fade(z);
	
	double nx00 = mix(n000, n100, u);
	double nx01 = mix(n001, n101, u);
	double nx10 = mix(n010, n110, u);
	double nx11 = mix(n011, n111, u);
	
	double nxy0 = mix(nx00, nx10, v);
	double nxy1 = mix(nx01, nx11, v);
	
	double nxyz = mix(nxy0, nxy1, w);
	
	if(nxyz < -1) nxyz = -1;
	if(nxyz > 1) nxyz = 1;
	
	return nxyz;
}

extern "C"
__global__ void sumOctave(int * output, int x, int y, int z, int width) {
	
	//int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	//int idx = threadIdx.x;
	//int idy = ((blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x);
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int pixelIndex = idx + (idy * width);
	
	x += idx;
	y += idy;

	double maxAmp = 0.0;
	double amp = 1.0;
	double freq = scale;
	double noise = 0;
	
	for(int i = 0; i < iterations; i++) {
		
		double adding = noise3D(x * freq, y * freq, z * freq) * amp;
		noise += adding;
		maxAmp += amp;
		amp *= persistance;
		freq *= 2;
		
	}
	
	noise /= maxAmp;
	
	if(noiseType == 1) {
	
		double xPeriod = 5.0;
		double yPeriod = 10.0;
		double tCoefficient = 5.0;
		
		noise = x * xPeriod / 900 + y * yPeriod / 600 + tCoefficient * noise;
		double sineVal = 256 * abs(sin(noise * PI));
		noise = (int)(sineVal) % 256;
	
	}
	else if(noiseType == 2) {
		
		if(noise <= 0) {
			noise = 0;
		}
		else {
			noise = 255;
		}
		
	}
	else {
		//noise = noise * (high - low) / 2 + (high + low) / 2;
		noise = map(noise, -1, 1, 0, 255);
	}
	
	output[pixelIndex] = noise;

}

extern "C"
__global__ void random(int * output, double x, double y, double z) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int pixelIndex = (idy * blockDim.y) + idx;
	
	x += idx;
	y += idy;
	
	output[pixelIndex] = (int)z % 256;

}
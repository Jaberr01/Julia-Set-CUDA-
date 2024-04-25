
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "EasyBMP.h"
#include "EasyBMP.cpp"

//Complex number definition
 struct Complex {	// typedef is not required for C++
	float x; 		// real part is represented on x-axis in output image
	float y; 		// imaginary part is represented by y-axis in output image
};

//Function declarations
void compute_julia(const char*, int, int);
void save_image(uchar4*, const char*, int, int);
__device__ Complex add(Complex, Complex);
__device__ Complex mul(Complex, Complex);
__device__ float mag(Complex);
__global__ void Kernel(uchar4* , int , int , float , float , float , float , float , float , int , int , Complex );

//main function
int main(void) {
	char* name = "test.bmp";
	compute_julia(name, 3000, 3000);	//width x height
	printf("Finished creating %s.\n", name);
	return 0;
}
__global__ void Kernel(uchar4* pixels, int width, int height, float w, float h, float x_min, float x_incr, float y_min, float y_incr, int max_iterations, int infinity, Complex c) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < width && col < height) {
	
		Complex z;
		z.x = x_min + col * x_incr;
		z.y = y_min + row * y_incr;
		//iteratively compute z = z^2 + c and check if z goes to infinity
		int n = 0;
		do {
			z = add(mul(z, z), c);								// z = z^2 + c
		} while (mag(z) < infinity && n++ < max_iterations);	// keep looping until z->infinity or we reach max_iterations

		// color each pixel based on above loop
		if (n == max_iterations) {								// if we reach max_iterations before z reaches infinity, pixel is black 
			pixels[col + row * width] = { 0,0,0,0 };
		}
		else {	
			unsigned char h1, h2, h3;
			double shift = 0.4;
			if (n <= shift*max_iterations / 3) {
				h1 = (unsigned char)(255 * sqrt((float)n / max_iterations));
				h2 = 25;
				h3 = 25;

			}
			else if (n <= 2 * shift * max_iterations / 3) {
				h1 = 25;
				h2 = (unsigned char)(255 * sqrt((float)n / max_iterations));
				h3 = 25;
			}
			else {
				h1 = 25;
				h2 = 25;
				h3 = (unsigned char)(255 * sqrt((float)n / max_iterations));
			}
			pixels[col + row * width] = { h1,h2,h3,255 };
		}
	}
		
	
}

// serial implementation of Julia set
void compute_julia(const char* filename, int width, int height) {
	//create output image
	unsigned int N = width * height;

	//PROBLEM SETTINGS (marked by '******')
	// **** Accuracy ****: lower values give less accuracy but faster performance
	int max_iterations = 400;
	int infinity = 20;													//used to check if z goes towards infinity

	// ***** Shape ****: other values produce different patterns. See https://en.wikipedia.org/wiki/Julia_set
	Complex c = { -0.8, 0.156 }; 										//the constant in z = z^2 + c

	// ***** Size ****: higher w means smaller size
	float w = 4;
	float h = w * height / width;										//preserve aspect ratio

	// LIMITS for each pixel
	float x_min = -w / 2, y_min = -h / 2;
	float x_incr = w / width, y_incr = h / height;

	uchar4* pixels;	//uchar4 is a CUDA type representing a vector of 4 chars
	cudaMallocManaged(&pixels, N * sizeof(uchar4));

	dim3 blockdim(32,32); //number of threads per block
	dim3 griddim((width + blockdim.x - 1) / blockdim.x, (height + blockdim.y - 1) / blockdim.y); //number of threads per block

	Kernel <<<griddim, blockdim >>> (pixels, width, height, w, h, x_min, x_incr, y_min, y_incr, max_iterations, infinity,c);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();


	//Write output image to a file (DO NOT parallelize this function)
	save_image(pixels, filename, width, height);

	//free memory
	cudaFree(pixels);
}

void save_image(uchar4* pixels, const char* filename, int width, int height) {
	BMP output;
	output.SetSize(width, height);
	output.SetBitDepth(24);
	// save each pixel to output image
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			uchar4 color = pixels[col + row * width];
			output(col, row)->Red = color.x;
			output(col, row)->Green = color.y;
			output(col, row)->Blue = color.z;
		}
	}
	output.WriteToFile(filename);
}

__device__ Complex add(Complex c1, Complex c2) {
	return{ c1.x + c2.x, c1.y + c2.y };
}

__device__ Complex mul(Complex c1, Complex c2) {
	return{ c1.x * c2.x - c1.y * c2.y, c1.x * c2.y + c2.x * c1.y };
}

__device__ float mag(Complex c) {
	return (float)sqrt((double)(c.x * c.x + c.y * c.y));
}

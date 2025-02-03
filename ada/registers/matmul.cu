#include <stdio.h>
#include "testbench.h"

#define BLOCKS_PER_SM 3 
#define LENGTH 50
#define TB_SIZE 512

__global__ void kernel(float *data, bool loop) {
	int a[LENGTH * LENGTH];
	int b[LENGTH * LENGTH];
	int c[LENGTH * LENGTH];
	int temp = 0;
	for (int i=0; i<LENGTH; i++) {
		for (int j=0; j<LENGTH; j++) {
			c[i*LENGTH + j] += a[i*LENGTH + j] * b[j*LENGTH + i];	
			temp += c[i*LENGTH + j];
		}
	}
	*data = temp;
}

int main() {
	// check which GPU is being used
	cudaDeviceProp deviceProp;
	SAFE(cudaGetDeviceProperties(&deviceProp, 0));
	printf("----- Device: %s -----\n", deviceProp.name);
	int sm_count = deviceProp.multiProcessorCount;
	printf(" SM Count: %d\n", sm_count);
  	float *data;
	//SAFE(cudaHostAlloc(&data, sizeof(float), cudaHostAllocMapped));
  	SAFE(cudaMalloc(&data, LENGTH*sizeof(float)));
  	SAFE(cudaMemset(data, 0, LENGTH*sizeof(float)));
	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record the start event
	cudaEventRecord(start);

  	//kernel<<<sm_count * BLOCKS_PER_SM, TB_SIZE>>>(data, false);
  	kernel<<<sm_count * BLOCKS_PER_SM, TB_SIZE>>>(data, false);
  	SAFE(cudaDeviceSynchronize());

	// Record the stop event
	cudaEventRecord(stop);
  	SAFE(cudaDeviceSynchronize());

	// Calculate the elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Output the elapsed time
	printf("----- Elapsed time: %f ms -----\n", milliseconds);
  	return 0;
}

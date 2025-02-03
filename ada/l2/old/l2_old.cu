#include <stdio.h>
#include "testbench.h"
#define L2_PCT 100.0
#define BLOCKS_PER_SM 1
#define SM 142
#define THREADS 1024

__global__ void vecInc(int *a, int n) {
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 173;
	for (int i=0; 1; i=(i+1)%173)
		a[idx + i] += 1;
}

int main() {
	// check which GPU is being used
	cudaDeviceProp deviceProp;
	SAFE(cudaGetDeviceProperties(&deviceProp, 0));
	printf("Device: %s\n", deviceProp.name);
	int l2Size = deviceProp.l2CacheSize;
	printf("L2 cache size: %d\n", l2Size);

	// check persistent portion
	size_t persistingL2CacheSize;
	SAFE(cudaDeviceGetLimit(&persistingL2CacheSize, cudaLimitPersistingL2CacheSize));
	printf("Previous persisting L2 cache size: %zu bytes\n", persistingL2CacheSize);

	// set persistent L2 size to 0
	SAFE(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));

	// verify persistent portion
	SAFE(cudaDeviceGetLimit(&persistingL2CacheSize, cudaLimitPersistingL2CacheSize));
	printf("New persisting L2 cache size: %zu bytes\n", persistingL2CacheSize);

	// array size
	int n = ( (int) (l2Size * (L2_PCT / 100.0) ) ) / 4;
	printf("Allocating array of size %d\n", n);

	// host memory
	int *h_a;
	h_a = (int *) malloc(n * sizeof(int));
	for (int i=0; i<n; i++) {
		h_a[i] = 1;
	}

	// device memory
	int *d_a;
	int bytes = n * sizeof(int);
	// if (bytes != l2Size) {
	// 	printf("Not allocating size of L2 cache\n");
	// 	free(h_a);
	// 	return 1;
	// }
	printf("Allocating %d bytes\n", bytes);
	SAFE(cudaMalloc(&d_a, bytes));
	SAFE(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record the start event
	cudaEventRecord(start);

	// launch kernel
	vecInc<<<SM * BLOCKS_PER_SM, THREADS>>>(d_a, n);
	SAFE(cudaDeviceSynchronize());
	
	// Record the stop event
	cudaEventRecord(stop);
	
	// Wait for the stop event to complete
	cudaEventSynchronize(stop);
	
	// Calculate the elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// Output the elapsed time
	printf("Elapsed time: %f ms\n", milliseconds);

	// print result
	SAFE(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
	printf("First ten:\n");
	for (int i=0; i<10; i++) {
		printf("%d\t", h_a[i]);
	}
	printf("\nLast range:\n");
	for (int i=25155583; i>25155573; i--) {
		printf("%d\t", h_a[i]);
	}
	printf("\nLast ten:\n");
	for (int i=n-1; i>n-10; i--) {
		printf("%d\t", h_a[i]);
	}
	printf("\n");

	// free memory
	SAFE(cudaFree(d_a));
	free(h_a);
	return 0;
}

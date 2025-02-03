#include <stdio.h>
#include <time.h>
#include "testbench.h"

#define BLOCKS_PER_SM 3
#define LENGTH 1000
#define TB_SIZE 512

__global__ void fillRegs(int *result, int n) {
	int arr[LENGTH];
	int arr2[LENGTH];
	int arr3[LENGTH];
	int arr4[LENGTH];
	int temp = 0;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	for (int i=0; i<n; i++) {
		arr[i] = n;
		temp += arr[i];
	}
	for (int i=0; i<n; i++) {
		arr2[i] = arr[i];
		temp += arr2[i];
	}
	for (int i=0; i<n; i++) {
		arr3[i] = arr2[i];
		arr3[i] += arr[i];
		temp += arr3[i];
	}
	for (int i=0; i<n; i++) {
		arr4[i] = arr[i] + arr3[i];
		arr4[i] += arr2[i];
		temp += arr4[i];

	}
	result[idx] = temp;
}


int main() {
	// check which GPU is being used
	cudaDeviceProp deviceProp;
	SAFE(cudaGetDeviceProperties(&deviceProp, 0));
	printf("----- Device: %s -----\n", deviceProp.name);
	int sm_count = deviceProp.multiProcessorCount;
	printf(" SM Count: %d\n", sm_count);

	// calculate total threads being launched
	int total_threads = TB_SIZE * BLOCKS_PER_SM * sm_count;

	// set persistent L2 size to 0
	SAFE(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));

	// verify persistent portion
	size_t persistingL2CacheSize;
	SAFE(cudaDeviceGetLimit(&persistingL2CacheSize, cudaLimitPersistingL2CacheSize));
	printf(" New persisting L2 cache size: %zu bytes\n", persistingL2CacheSize);

	// benchmark info
	printf("-----Allocating int array of length %d -----\n", LENGTH);
	printf(" Array is %lu bytes\n", LENGTH * sizeof(int));
	printf(" Launching %d threads\n", total_threads);

	// accumulation array for ensuring array is used
	int *result;
	SAFE(cudaHostAlloc(&result, total_threads*sizeof(int), cudaHostAllocMapped));
	for (int i=0; i<total_threads; i++)
		result[i] = 0;

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record the start event
	cudaEventRecord(start);

	// launch kernel
	fillRegs<<<sm_count * BLOCKS_PER_SM, TB_SIZE>>>(result, LENGTH);
	SAFE(cudaDeviceSynchronize());
	
	// Record the stop event
	cudaEventRecord(stop);
	
	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// Calculate the elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Output the elapsed time
	printf("----- Elapsed time: %f ms -----\n", milliseconds);
	printf("First ten: ");
	for (int i=0; i<10; i++) {
		printf("%d\t", result[i]);
	}
	printf("\nLast ten: ");
	for (int i=total_threads-1; i>total_threads-10; i--) {
		printf("%d\t", result[i]);
	}
	printf("\n");

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}

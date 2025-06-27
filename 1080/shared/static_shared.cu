#include <stdio.h>
#include "testbench.h"

#define BLOCKS_PER_SM 1
#define NUMBER 8
#define SHARED_PCT 51.51
#define SHARED_MEM_TB (int) (101376.0 * (SHARED_PCT / 100.0))
#define ELEMENTS SHARED_MEM_TB / 4
#define SM 142
#define STATIC_ELEMENTS 12288

__global__ void fillShared(int *blockCounter, int *flag) {
	__shared__ int static_array[STATIC_ELEMENTS];
	extern __shared__ float shared_array[];
	for (int i=0; i<ELEMENTS; i++) {
		shared_array[i] = NUMBER;
	}
	for (int i=0; i<STATIC_ELEMENTS; i++) {
		static_array[i] = NUMBER;
	}
	shared_array[0] += static_array[8];

	// ensure all blocks have loaded their portion of shared memory
	atomicAdd(blockCounter, 1);

	// ensure updated blockCounter is visible across blocks
	__threadfence();

	// tell CPU that shared memory is fully saturated
	if (blockIdx.x == 0) {
		while (*blockCounter < gridDim.x) {
			// wait for all blocks to load shared memory
			__threadfence();
		}
		__threadfence();
		*flag = 0;
	}

	// spin with desired shared memory usage
	while (0) { }
}

int main() {
	// Check the opt in value for dynamic shared memory
	cudaDeviceProp prop;
	int device = 0;
	SAFE(cudaGetDeviceProperties(&prop, device));
	printf("Device: %s\nMax shared memory size opt in: %lu\n", prop.name, prop.sharedMemPerBlockOptin);

	// print shared array size per TB/SM
	printf("Every TB is addressing a shared memory array of size %d\n", ELEMENTS);

	// flag for CPU synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// block counter for signaing when shared memory caches are saturated
	int *blockCounter;
	SAFE(cudaMalloc(&blockCounter, sizeof(int)));
	SAFE(cudaMemset(blockCounter, 0, sizeof(int)));

	// Allow TB to address dynamic shared memory as needed
	printf("Setting TB shared memory max to %d bytes\n", SHARED_MEM_TB);
	SAFE(cudaFuncSetAttribute(fillShared, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM_TB));

	// Adjust shared partition to match usage
	SAFE(cudaFuncSetAttribute(fillShared, cudaFuncAttributePreferredSharedMemoryCarveout, SHARED_PCT));

	// Confirm shared carevout
	cudaFuncAttributes attr;
	SAFE(cudaFuncGetAttributes(&attr, fillShared));
	printf("Carveout set to %d%% \n", attr.preferredShmemCarveout);
	printf("Static shared memory size: %lu\n", attr.sharedSizeBytes);

	// launch kernel and spin
	fillShared<<<SM * BLOCKS_PER_SM, 1, SHARED_MEM_TB>>>(blockCounter, flag);
	printf("Kernel launched - waiting for cache saturation...\n");
	while (*flag) { /* wait for caches to saturate */ }
	printf("Shared memory caches are saturated!\n");
	SAFE(cudaDeviceSynchronize());

	return 0;
}



#include <stdio.h>
#include "testbench.h"

#define BLOCKS_PER_SM 1
#define NUMBER 8
#define SHARED_PCT 75.0
#define SHARED_MEM_TB (int) (101376.0 * (SHARED_PCT / 100.0))
#define ELEMENTS SHARED_MEM_TB / 4
#define SM 142

__global__ void fillShared(int *blockCounter, int *flag) {
	extern __shared__ float shared_array[];
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
	while (1) { }
}

int main() {
	// Ensure the opt in value is correct
	cudaDeviceProp prop;
	int device = 0;
	SAFE(cudaGetDeviceProperties(&prop, device));
	printf("Device: %s\nMax shared memory size opt in: %lu\n", prop.name, prop.sharedMemPerBlockOptin);

	// flag for CPU synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// block counter for signaing when shared memory caches are saturated
	int *blockCounter;
	SAFE(cudaMalloc(&blockCounter, sizeof(int)));
	SAFE(cudaMemset(blockCounter, 0, sizeof(int)));

	// Adjust shared partition to match usage
	SAFE(cudaFuncSetAttribute(fillShared, cudaFuncAttributePreferredSharedMemoryCarveout, SHARED_PCT));

	// Confirm shared carevout
	cudaFuncAttributes attr;
	SAFE(cudaFuncGetAttributes(&attr, fillShared));
	printf("Carveout set to %d%% \n", attr.preferredShmemCarveout);

	// launch kernel and spin
	fillShared<<<SM * BLOCKS_PER_SM, 1, SHARED_MEM_TB>>>(blockCounter, flag);
	printf("Kernel launched - waiting for cache saturation...\n");
	while (*flag) { /* wait for caches to saturate */ }
	printf("Shared memory caches are saturated!\n");
	SAFE(cudaDeviceSynchronize());

	return 0;
}



#include <stdio.h>
#include "testbench.h"

#define BLOCKS 10
#define DEBUG 1
#define ELEMENTS 12288
#define NUMBER 8
#define SHARED_MEM_TB 49152

__global__ void fillShared(int *blockCounter, int *flag, int *totals) {
	extern __shared__ float shared_array[];
	for (int i=0; i<ELEMENTS; i++) {
		shared_array[i] = NUMBER;
	}

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
		*flag = 0;
	}

	// spin with desired shared memory usage
	while (0) { }
	totals[blockIdx.x] = shared_array[ELEMENTS - 1];
}

int
main()
{
	// flag for CPU synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// block counter for signaing when shared memory caches are saturated
	int *blockCounter;
	SAFE(cudaMalloc(&blockCounter, sizeof(int)));
	SAFE(cudaMemset(blockCounter, 0, sizeof(int)));

	// host array for checking if all blocks filled their shared caches with NUMBER
	int *h_all;
	SAFE(cudaHostAlloc(&h_all, ELEMENTS * BLOCKS * sizeof(int), cudaHostAllocMapped));
	memset(h_all, 0, ELEMENTS * BLOCKS * sizeof(int));

	// launch kernel and spin
	fillShared<<<BLOCKS, 1, SHARED_MEM_TB>>>(blockCounter, flag, h_all);
	printf("Kernel launched - waiting for cache saturation...\n");
	while (*flag) { /* wait for caches to saturate */ }
	printf("Shared memory caches are saturated!\n");
	SAFE(cudaDeviceSynchronize());

#ifdef DEBUG
	int total = 0;
	for (int i=0; i<BLOCKS; i++) {
		total += h_all[i];
	}

	printf("Expected total: %d\n", BLOCKS * NUMBER);
	printf("Actual total: %d\n", total);
	if (BLOCKS * NUMBER == total)
		printf("Totals match!\n");
	else
		printf("Totals do not match :(\n");
#endif
	return 0;
}



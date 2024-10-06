#include <stdio.h>
#include "testbench.h"
#define CONSTANT_PCT 100.0
#define CONSTANT_MEMORY (int) (65536.0 * (CONSTANT_PCT / 100.0))
#define LENGTH CONSTANT_MEMORY / 4

__constant__ int constantData[LENGTH];

__global__ void readConstant(int *result, int *flag) {
	int sum = 0;
	for (int i=0; i<LENGTH; i++) {
		sum += constantData[i];
	}
	*result = sum;
	*flag = 0;
	while (0) { }
}

int
main()
{
	// print cache percent info
	printf("Constant cache usage: %.1f | %d bytes | %d elements\n", CONSTANT_PCT, CONSTANT_MEMORY, LENGTH);

	// allocate host data for constant cache
	int hostData[LENGTH];
	for (int i=0; i<LENGTH; i++) {
		hostData[i] = 2;
	}

	// copy data to constant memory cache
	SAFE(cudaMemcpyToSymbol(constantData, hostData, CONSTANT_MEMORY));

	// flag for synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// result ptr for sum
	int *result;
	SAFE(cudaHostAlloc(&result, sizeof(int), cudaHostAllocMapped));
	*result = 0;

	// spin on GPU once constant memory is accessed
	readConstant<<<1, 1>>>(result, flag);
	while (*flag) {}
	printf("Constant memory has been read!\n");
	SAFE(cudaDeviceSynchronize());

	// print result
	printf("Total: %d\n", *result);

	return 0;
}

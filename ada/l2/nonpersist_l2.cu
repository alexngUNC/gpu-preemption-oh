#include <stdio.h>
#include "testbench.h"
#define L2_PCT 10.0

__global__ void vecInc(int *a, int n, int *flag) {
	for (int i=0; i<n; i++) {
		a[i] += 1;
	}
	__threadfence();
	*flag = 0;
	while (0) { /* spin once saturated */ }
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

	// flag for CPU synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// launch kernel
	vecInc<<<1, 1>>>(d_a, n, flag);
	while (*flag) { /* wait for caches to saturate */ }
	printf("Caches are saturated! Spinning...\n");
	SAFE(cudaDeviceSynchronize());

	// print result
	SAFE(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
	printf("First ten:\n");
	for (int i=0; i<10; i++) {
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

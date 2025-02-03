#include <stdio.h>
#include <time.h>
#include "testbench.h"

#define BLOCKS_PER_SM 3
#define ELTS_PER_THREAD 270
#define L1_SIZE 131072
#define TB_SIZE 512

//#define DEBUG 1

__global__ void vecInc(int *a, int n) {
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELTS_PER_THREAD;

#ifdef DEBUG
	for (int i=0; i<ELTS_PER_THREAD; i++)
		a[idx + i] = idx;
#else
	for (int i=0; 1; i=(i+1) % ELTS_PER_THREAD)
		a[idx + i] = idx;
#endif

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

	// check persistent L2 portion
	int l2Size = deviceProp.l2CacheSize;
	printf(" L2 cache size: %d bytes\n", l2Size);
	int cache_total = l2Size + L1_SIZE * sm_count;
	printf(" Total amount of memory stored in L1 and L2 caches: %d bytes\n", cache_total);
	size_t persistingL2CacheSize;
	SAFE(cudaDeviceGetLimit(&persistingL2CacheSize, cudaLimitPersistingL2CacheSize));
	printf(" Previous persisting L2 cache size: %zu bytes\n", persistingL2CacheSize);

	// set persistent L2 size to 0
	SAFE(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));

	// verify persistent portion
	SAFE(cudaDeviceGetLimit(&persistingL2CacheSize, cudaLimitPersistingL2CacheSize));
	printf(" New persisting L2 cache size: %zu bytes\n", persistingL2CacheSize);

	// array size
	int n = total_threads * ELTS_PER_THREAD;
	int bytes = n * sizeof(int);
	float overflow_ratio = ((float)(bytes)) / ((float)(cache_total));
	printf("-----Allocating int array of length %d -----\n", n);
	printf(" Array is %d bytes\n", bytes);
	printf(" Launching %d threads\n", total_threads);
	printf(" Each thread will access %d elements\n", ELTS_PER_THREAD);
	printf(" Array is %f times the total amount of memory stored in the L1 and L2 caches\n", overflow_ratio);


	// seed the random number generator
    	srand(time(NULL));

	// host memory
	int *h_a;
	h_a = (int *) malloc(bytes);
	for (int i=0; i<n; i++) {
		h_a[i] = rand();
	}

	// device memory
	int *d_a;
	if (bytes < cache_total) {
		printf("Not allocating enough memory to saturate all caches. Exiting...\n");
		free(h_a);
		return 1;
	}
	printf(" Allocating %d bytes on GPU\n", bytes);
	SAFE(cudaMalloc(&d_a, bytes));
	SAFE(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record the start event
	cudaEventRecord(start);

	// launch kernel
	vecInc<<<sm_count * BLOCKS_PER_SM, TB_SIZE>>>(d_a, n);
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
	printf("----- Elapsed time: %f ms -----\n", milliseconds);

	// print result
	SAFE(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
	printf("First ten: ");
	for (int i=0; i<10; i++) {
		printf("%d\t", h_a[i]);
	}
	printf("\nLast ten: ");
	for (int i=n-1; i>n-10; i--) {
		printf("%d\t", h_a[i]);
	}
	printf("\n");

	// free memory
	SAFE(cudaFree(d_a));
	free(h_a);
	return 0;
}

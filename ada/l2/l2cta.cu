#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include "testbench.h"

#define BLOCKS_PER_SM 1
#define ITERATIONS 200
#define L1_SIZE 131072
#define NUM_BLOCKS 1
#define STRIDE 32
#define TB_SIZE 1024
#define TOTAL_BUFFER_LENGTH 131072

#define DEBUG 1

__global__ void vecStore(int *a, size_t total_buffer_length) {
	size_t portion_each_thread_covers = total_buffer_length / blockDim.x;
	size_t start_index = portion_each_thread_covers * threadIdx.x;
	size_t end_index = start_index + portion_each_thread_covers;
	for (int i=0; i<ITERATIONS; i++) {
		for (size_t i = start_index; i < end_index; i += STRIDE) {
			a[i] += i;
		}
	}
}

__global__ void vecStore2(int *a, size_t total_buffer_length) {
	size_t portion_each_thread_covers = total_buffer_length / blockDim.x;
	size_t start_index = portion_each_thread_covers * threadIdx.x;
	size_t end_index = start_index + portion_each_thread_covers;
	for (int i=0; i<ITERATIONS; i++) {
		for (size_t i = start_index; i < end_index; i += STRIDE) {
			a[i] += i;
		}
	}
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

	// Set shared carveout to 0
	printf(" Setting shared memory carveout to 0\n");
	SAFE(cudaFuncSetAttribute(vecStore, cudaFuncAttributePreferredSharedMemoryCarveout, 0));

	// array size
	int n = TOTAL_BUFFER_LENGTH;
	int bytes = n * sizeof(int);
	float l1_ratio = ((float)(bytes)) / ((float)(L1_SIZE));
	float l2_ratio = ((float)(bytes)) / ((float)(l2Size));
	float cache_total_ratio = ((float)(bytes)) / ((float)(cache_total));
	printf("----- Allocating int array of length %d -----\n", n);
	printf(" Array is %d bytes\n", bytes);
	printf(" Launching %d threads\n", total_threads);
	printf(" Array is %f times the total amount of memory stored in the L1 cache\n", l1_ratio);
	printf(" Array is %f times the total amount of memory stored in the L2 cache\n", l2_ratio);
	printf(" Array is %f times the total amount of memory stored in the L1 and L2 caches\n", cache_total_ratio);


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
	int *d_a2;
	if (bytes < cache_total) {
		//printf("Not allocating enough memory to saturate all caches. Exiting...\n");
		//free(h_a);
		//return 1;
		printf("Warning: not allocating enough memory to saturate L2 & all L1 caches\n");
	}
	printf(" Allocating %d bytes on GPU\n", bytes*2);
	SAFE(cudaMalloc(&d_a, bytes));
	SAFE(cudaMalloc(&d_a2, bytes));
	SAFE(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
	SAFE(cudaMemcpy(d_a2, h_a, bytes, cudaMemcpyHostToDevice));

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Create a context so benchmark is not the last thing to run
	CUdevice device;
	SAFE_D(cuDeviceGet(&device, 0));
	CUcontext dummy_ctx;
	SAFE_D(cuCtxCreate(&dummy_ctx, 0, device));
	SAFE_D(cuCtxDestroy(dummy_ctx));

	// Wait for keyboard input so preemption mode can be changed
	printf("Press enter to continue\n");
	fgetc(stdin);
	printf("Launching kernel...\n");

	// Record the start event
#ifdef DEBUG
	cudaEventRecord(start);
#endif

	// launch kernel
	vecStore<<<sm_count * BLOCKS_PER_SM, TB_SIZE>>>(d_a, n);
	vecStore2<<<sm_count * BLOCKS_PER_SM, TB_SIZE>>>(d_a2, n);
	printf("GPU is spinning!\n");
	SAFE(cudaDeviceSynchronize());
	
	// Record the stop event
#ifdef DEBUG
	cudaEventRecord(stop);
	
	// Wait for the stop event to complete
	cudaEventSynchronize(stop);
#endif

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

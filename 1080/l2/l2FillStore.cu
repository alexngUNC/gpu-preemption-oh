#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "task_host_utilities.cu"
#include "testbench.h"

#define BLOCKS_PER_SM 1
#define L1_SIZE 49152
#define STRIDE 32
#define TB_SIZE 1024
#define TOTAL_BUFFER_LENGTH 49152

#define MIN_PREEMPT_TICKS 20*1000 // 20us

//#define DEBUG 1


__global__ void vecIncrement(int *a, size_t total_buffer_length, int num_preemptions) {
	size_t portion_each_thread_covers = total_buffer_length / blockDim.x;
	size_t start_index = portion_each_thread_covers * threadIdx.x;
	size_t end_index = start_index + portion_each_thread_covers;
	int preempt_count = 0;
#ifndef DEBUG
	while (preempt_count < num_preemptions) {
#endif
		// saturate cache
		for (size_t i = start_index; i < end_index; i += STRIDE) {
			a[i] += i;
		}

#ifndef DEBUG
		// spin until preempted
		uint64_t last_time = GlobalTimer64();
		while (1) {
			uint64_t now = GlobalTimer64();
			if (now > last_time + MIN_PREEMPT_TICKS) {
				preempt_count += 1;
				break;
			}
			last_time = now;
		}
	}
#endif
}


static const char* usage_msg = "\
Usage: %s NUM_PREEMPTIONS\n\
Saturate L1 caches -> spin until preempted -> repeat.\n\
  NUM_PREEMPTIONS GPU kernel repeats until this many preemptions have occurred.\n";

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, usage_msg, argv[0]);
		return 1;
	}
	int num_preemptions = atoi(argv[1]);
	if (!num_preemptions) {
		fprintf(stderr, usage_msg, argv[0]);
		return 1;
	}

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

	// Set shared carveout to 0
	printf(" Setting shared memory carveout to 0\n");
	SAFE(cudaFuncSetAttribute(vecIncrement, cudaFuncAttributePreferredSharedMemoryCarveout, 0));

	// array size
	size_t n = TOTAL_BUFFER_LENGTH;
	int bytes = n * sizeof(int);
	float l1_ratio = ((float)(bytes)) / ((float)(L1_SIZE));
	float l2_ratio = ((float)(bytes)) / ((float)(l2Size));
	float cache_total_ratio = ((float)(bytes)) / ((float)(cache_total));
	printf("----- Allocating int array of length %zu -----\n", n);
	printf(" Array is %d bytes\n", bytes);
    printf(" Launching %d blocks with %d threads in each\n", sm_count * BLOCKS_PER_SM, TB_SIZE);
	printf(" Launching %d total threads\n", total_threads);
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
	if (bytes < cache_total) {
		//printf("Not allocating enough memory to saturate all caches. Exiting...\n");
		//free(h_a);
		//return 1;
		printf("Warning: not allocating enough memory to saturate L2 & all L1 caches\n");
	}
	printf(" Allocating %d bytes on GPU\n", bytes);
	SAFE(cudaMalloc(&d_a, bytes));
	SAFE(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel launch config
	int maxBlocksPerSm = 0;
	int numThreads = TB_SIZE;
	SAFE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSm, vecIncrement, numThreads, 0));
	printf("Max blocks per SM: %d\n", maxBlocksPerSm);
	if (sm_count * BLOCKS_PER_SM > maxBlocksPerSm * sm_count) {
		fprintf(stderr, "Attempting to launch more blocks than can be resident on the SMs. Terminating.\n");
		return 1;
	}
	//dim3 dimGrid(sm_count * BLOCKS_PER_SM, 1, 1);
	//dim3 dimBlock(numThreads, 1, 1);
	//void *kernelArgs[] = { &d_a, &n, &num_preemptions };
	// wait for keyboard input to start
	printf("Press any key to continue.\n");
	fgetc(stdin);
	printf("Launching kernel...\n");
	// Record the start event
	cudaEventRecord(start);
	// launch kernel
	vecIncrement<<<sm_count * BLOCKS_PER_SM, TB_SIZE>>>(d_a, n, num_preemptions);
	//SAFE(cudaLaunchCooperativeKernel((void *) vecIncrement, dimGrid, dimBlock, kernelArgs, 0, 0)); 
	printf("Kernel launched!\n");
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

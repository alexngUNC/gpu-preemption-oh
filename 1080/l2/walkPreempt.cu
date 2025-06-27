#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "five_number_summary.h"
#include "task_host_utilities.cu"
#include "testbench.h"

#define BLOCKS_PER_SM 1
#define L1_SIZE 49152
#define STRIDE 32
#define TB_SIZE 1
// saturate 4 times: 49152
// L1: 12288
// 1 read: 1024
//#define TOTAL_BUFFER_LENGTH 6291456
#define TOTAL_BUFFER_LENGTH 49152

#define MIN_PREEMPT_TICKS 20*1000 // 20us

//#define DEBUG 1

struct interval {
	uint64_t duration;
};


__global__ void vecRead(int *a, size_t total_buffer_length, int num_preemptions, struct interval *ivls) {
	struct interval *curr_ivl = ivls;
	int preempt_count = 0;
#ifndef DEBUG
	while (preempt_count < num_preemptions) {
#endif
		// saturate cache
		uint64_t start = GlobalTimer64();
		for (size_t i = 0; i < total_buffer_length; i += STRIDE) {
			a[i] += i;
		}
		uint64_t end = GlobalTimer64();
		curr_ivl->duration = end - start;
		curr_ivl++;

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
  Walk buffer -> spin until preempted -> repeat.\n\
  NUM_PREEMPTIONS GPU kernel walks buffer for this many preemptions/iterations.\n\
  Logs time to walk buffer each time.\n";

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
	//int total_threads = TB_SIZE * BLOCKS_PER_SM * sm_count;
	int total_threads = 1;

	// check persistent L2 portion
	int l2Size = deviceProp.l2CacheSize;
	printf(" L2 cache size: %d bytes\n", l2Size);
	int cache_total = l2Size + L1_SIZE * sm_count;
	printf(" Total amount of memory stored in L1 and L2 caches: %d bytes\n", cache_total);

	// Set shared carveout to 0
	printf(" Setting shared memory carveout to 0\n");
	SAFE(cudaFuncSetAttribute(vecRead, cudaFuncAttributePreferredSharedMemoryCarveout, 0));

	// array size
	size_t n = TOTAL_BUFFER_LENGTH;
	int bytes = n * sizeof(int);
	float l1_ratio = ((float)(bytes)) / ((float)(L1_SIZE));
	float l2_ratio = ((float)(bytes)) / ((float)(l2Size));
	float cache_total_ratio = ((float)(bytes)) / ((float)(cache_total));
	printf("----- Allocating int array of length %zu -----\n", n);
	printf(" Array is %d bytes\n", bytes);
	//printf(" Launching %d blocks with %d threads in each\n", sm_count * BLOCKS_PER_SM, TB_SIZE);
	printf(" Launching %d blocks with %d threads in each\n", 1, 1);
	printf(" Launching %d total threads\n", total_threads);
	printf(" Array is %f times the total amount of memory stored in the L1 cache\n", l1_ratio);
	printf(" Array is %f times the total amount of memory stored in the L2 cache\n", l2_ratio);
	printf(" Array is %f times the total amount of memory stored in the L1 and L2 caches\n", cache_total_ratio);

	// host memory
	int *h_a;
	printf(" Allocating %d bytes on GPU\n", bytes);
	h_a = (int *) malloc(bytes);
	if (h_a == NULL) {
		fprintf(stderr, "Could not allocate host memory\n");
		return 1;
	}
	if (bytes < cache_total) {
		//printf("Not allocating enough memory to saturate all caches. Exiting...\n");
		//free(h_a);
		//return 1;
		printf("Warning: not allocating enough memory to saturate L2 & all L1 caches\n");
	}
	// seed the random number generator
	srand(time(NULL));
	for (int i=0; i<n; i++) {
		h_a[i] = rand();
	}
	// device memory
	int *d_a;
	SAFE(cudaMalloc(&d_a, bytes));
	SAFE(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel launch config
	int maxBlocksPerSm = 0;
	int numThreads = TB_SIZE;
	SAFE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSm, vecRead, numThreads, 0));
	printf("Max blocks per SM: %d\n", maxBlocksPerSm);
	if (sm_count * BLOCKS_PER_SM > maxBlocksPerSm * sm_count) {
		fprintf(stderr, "Attempting to launch more blocks than can be resident on the SMs. Terminating.\n");
		return 1;
	}

	// interval logging
	struct interval *ivls_gpu, *ivls;
	SAFE(cudaMalloc(&ivls_gpu, num_preemptions * sizeof(struct interval)));
	ivls = (struct interval *) malloc(num_preemptions * sizeof(struct interval));
	if (!ivls) {
		perror("While allocating interval host memory");
		return 1;
	}

	// wait for keyboard input to start
	printf("Press any key to continue.\n");
	fgetc(stdin);
	printf("Launching kernel...\n");
	// Record the start event
	cudaEventRecord(start);
	// launch kernel
	vecRead<<<1, 1>>>(d_a, n, num_preemptions, ivls_gpu);
	printf("Kernel launched!\n");
	SAFE(cudaDeviceSynchronize());
	// Record the stop event
	cudaEventRecord(stop);
	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// copy ivl data back
	SAFE(cudaMemcpy(ivls, ivls_gpu, num_preemptions * sizeof(struct interval), cudaMemcpyDeviceToHost));

	// Calculate the elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// Output the elapsed time
	printf("----- Elapsed time: %f ms -----\n", milliseconds);

	// print result
	num_preemptions = (num_preemptions <= 10) ? num_preemptions : 10;
	for (int i=0; i<num_preemptions; i++) {
		printf("%lu ", ivls[i].duration);
	}
	printf("\n");
	fiveNumberSummary((uint64_t *) ivls, num_preemptions);

	// free memory
	SAFE(cudaFree(d_a));
	return 0;
}

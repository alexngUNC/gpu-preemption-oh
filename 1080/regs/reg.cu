#include <stdio.h>
#include <time.h>

#include "task_host_utilities.cu"
#include "testbench.h"

#define BLOCKS_PER_SM 2
#define LENGTH 85
#define TB_SIZE 1024


#define MIN_PREEMPT_TICKS 20*1000 // 20us

#define DEBUG 1

__global__ void loadRegs(int *data, bool printCond) {
	int sdata[LENGTH];
	int preempt_count = 0;

#ifndef DEBUG
	while (preempt_count < num_preemptions) {
#endif
		// load registers
		for (int i = 0; i < LENGTH; i++) {
			sdata[i] = data[i];
		}
		for (int i=0; i<LENGTH; i++) {
			if (printCond) printf("sdata[%d] = %d\n", i, sdata[i]);
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
Saturate registers -> spin until preempted -> repeat.\n\
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

	// alloc device memory
	int *data;
	//SAFE(cudaHostAlloc(&data, sizeof(int), cudaHostAllocMapped));
	int bytes = LENGTH * sizeof(int);
	printf("Allocating %d bytes on the GPU\n", bytes);
	SAFE(cudaMalloc(&data, bytes));
	SAFE(cudaMemset(data, 0, bytes));

	// Create CUDA events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// wait for user input before launching kernel
	printf("Press any key to continue.\n");
	fgetc(stdin);
	printf("Launching kernel...\n");

	// Record the start event
	cudaEventRecord(start);

	// launch kernel
	loadRegs<<<sm_count * BLOCKS_PER_SM, TB_SIZE>>>(data, false);
	printf("Kernel launched!\n");
	SAFE(cudaDeviceSynchronize());

	// Record the stop event
	cudaEventRecord(stop);
	SAFE(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);

	// Calculate the elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Output the elapsed time
	printf("----- Elapsed time: %f ms -----\n", milliseconds);
	return 0;
}

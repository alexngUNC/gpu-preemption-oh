/* 
 * Simple kernel that spins on the GPU for a specified number of iterations,
 * while tracking and printing the necessary CPU time.
 */
#include <time.h>
#include <stdio.h>
#include <cuda.h>

#include "task_host_utilities.cu"
#include "testbench.h"

#define TIMESLICE 1.131*1000000 // 1.131ms
#define EPSILON 0.035*1000000 // time difference from timeslice end
#define MAX_PAGES  100 // TODO: change for jetson orin total 4 KB pages for RTX 3060 Ti
#define PAGE_SIZE 4096

__global__ void loop_on_gpu(char *page, int *flag) {
	bool faulted = false;
	uint64_t start = GlobalTimer64();
	// TODO: change kernel so it continues for a bit after the page fault
	while (!faulted) {
		uint64_t now = GlobalTimer64();
		if (now > start + TIMESLICE - EPSILON) {
			// timeslice is about to expire so page fault
			*page = 0;
			faulted = true;
		}
	}
	// flag for synchronization
	*flag = 0;
}

int main(int argc, char **argv) {
	int res, *__unused;
	struct timespec start, end;

	if (argc != 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
		fprintf(stderr, "Usage: %s <# of millions of iterations, or -1 for infinite>\n",
		        argv[0]);
		return 1;
	}

	// Input is multiplied by one million, unless infinite
	unsigned long num_iters = strtoul(argv[1], NULL, 10);
	// if (num_iters != (unsigned long)(-1))
	// 	num_iters *= 1000 * 1000;

	// Initialize CUDA and a context (hack)
	SAFE(cudaMalloc(&__unused, 8));

	// allocate memory for pages
	if (num_iters >= MAX_PAGES) {
		fprintf(stderr, "Not enough pages to fulfill interval count. Aborting...\n");
		return 1;
	}
	CUdeviceptr d_pages;
	SAFE_D(cuMemAllocManaged(&d_pages, num_iters * PAGE_SIZE, CU_MEM_ATTACH_GLOBAL));

	// get gpu device to force migration
	CUdevice device;
	SAFE_D(cuDeviceGet(&device, 0));

	// flag for synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// Run iterations on a single thread
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (size_t i=0; i<num_iters; i++) {
		// Make sure it is resident in host memory so it triggers a page fault
		SAFE_D(cuMemAdvise(d_pages + i * PAGE_SIZE, PAGE_SIZE, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, CU_DEVICE_CPU));
		*(((char *) d_pages) + i * PAGE_SIZE) =  1;
		 
		// Advise the pages to be on the GPU so they migrate
		SAFE_D(cuMemAdvise(d_pages + i * PAGE_SIZE, PAGE_SIZE, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device));

		// launch page fault kernel
		loop_on_gpu<<<1,1>>>((char *) d_pages + i * PAGE_SIZE, flag);

		// busy wait until kernel finishes
		while (*flag);

	}
	SAFE(cudaDeviceSynchronize());
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	// Print detailed timing information
	long elapsed = timediff(start, end);
	fprintf(stderr, "Started at %ld ns, ended at %ld ns\n",
	        s2ns(start.tv_sec) + start.tv_nsec, s2ns(end.tv_sec) + end.tv_nsec);
	fprintf(stderr, "%ld ns (%.2f ms) elapsed\n", elapsed, elapsed / (1000 * 1000.));

	// Verify success (fool optimizer)
	//SAFE(cudaMemcpy(&res, __unused, 8, cudaMemcpyDeviceToHost));
	// Theoretically this can happen if `__unused` wraps around; maybe for a
	// very small `long` type, or a very long run. More likely indicates an error.
	// if (!res)
	//		fprintf(stderr, "CRITICAL: Zero iterations seem completed. Likely incorrect "
	//		        "arguments, internal error, or corruption of CUDA internal state.\n");

	return 0;
}

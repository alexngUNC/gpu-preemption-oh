#include "testbench.h"
#include <stdio.h>
__global__ void spin() {
	while (1) {}
}

int main() {
	spin<<<1, 1>>>();
	SAFE(cudaDeviceSynchronize());
	return 0;
}

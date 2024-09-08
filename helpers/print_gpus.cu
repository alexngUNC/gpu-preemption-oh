#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(error));
        return -1;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, i);

        if (error != cudaSuccess) {
            fprintf(stderr, "Error getting device properties for device %d: %s\n", i, cudaGetErrorString(error));
            return -1;
        }

        printf("Device %d: %s\n", i, deviceProp.name);
	printf("Total Global memory: %zu\n", deviceProp.totalGlobalMem);
    }

    return 0;
}


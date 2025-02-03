#include <stdio.h>
#define SSS 247
#define WITH_PRINTF 1

template <typename T>
__global__ void kernel(T *data){

  T sdata[SSS];
#pragma unroll
  for (int i = 0; i < SSS; i++)
    sdata[i] = data[i];

  T sum = 0.0f;
#ifdef WITH_PRINTF
#pragma unroll
  for (int i = 0; i < SSS; i++)
    printf("sdata[%d] = %f\n", i, sdata[i]);
#endif
#pragma unroll
  for (int i = 0; i < SSS; i++)
    sum += sdata[i];

  printf("sum = %f\n", sum);
}

int main() {
  float *data;
  cudaMalloc(&data, SSS*sizeof(float));
  cudaMemset(data, 0, SSS*sizeof(float));
  kernel<<<1,1>>>(data);
  cudaDeviceSynchronize();
  return 0;
}

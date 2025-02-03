#include <stdio.h>
#include "testbench.h"

#define BLOCKS_PER_SM 1
#define DATA 8
#define REG_PCT 100.0
#define ELEMENTS (int) (65536 * (REG_PCT / 100.0))
#define SM 142

__global__ void fillRegs(int *result) {
	/*
	int reg_array[ELEMENTS];
	for (int i=0; i<ELEMENTS; i++)
		reg_array[i] = DATA;
	for (int i=0; i<ELEMENTS; i++)
		reg_array[i] += reg_array[i];
	*/
	volatile int a1 = 1234;
	volatile int a2 = 5678;
	volatile int a3 = 2345;
	volatile int a4 = 6789;
	volatile int a5 = 3456;
	volatile int a6 = 7890;
	volatile int a7 = 4567;
	volatile int a8 = 1234;
	volatile int a9 = 5678;
	volatile int a10 = 2345;
	volatile int a11 = 6789;
	volatile int a12 = 3456;
	volatile int a13 = 7890;
	volatile int a14 = 4567;
	volatile int a15 = 1234;
	volatile int a16 = 5678;
	volatile int a17 = 2345;
	volatile int a18 = 6789;
	volatile int a19 = 3456;
	volatile int a20 = 7890;
	volatile int a21 = 4567;
	volatile int a22 = 1234;
	volatile int a23 = 5678;
	volatile int a24 = 2345;
	volatile int a25 = 6789;
	volatile int a26 = 3456;
	volatile int a27 = 7890;
	volatile int a28 = 4567;
	volatile int a29 = 1234;
	volatile int a30 = 5678;
	volatile int a31 = 2345;
	volatile int a32 = 6789;
	volatile int a33 = 3456;
	volatile int a34 = 7890;
	volatile int a35 = 4567;
	volatile int a36 = 1234;
	volatile int a37 = 5678;
	volatile int a38 = 2345;
	volatile int a39 = 6789;
	volatile int a40 = 3456;
	volatile int a41 = 7890;
	volatile int a42 = 4567;
	volatile int a43 = 1234;
	volatile int a44 = 5678;
	volatile int a45 = 2345;
	volatile int a46 = 6789;
	volatile int a47 = 3456;
	volatile int a48 = 7890;
	volatile int a49 = 4567;
	volatile int a50 = 1234;
	volatile int a51 = 5678;
	volatile int a52 = 2345;
	volatile int a53 = 6789;
	volatile int a54 = 3456;
	volatile int a55 = 7890;
	volatile int a56 = 4567;
	volatile int a57 = 1234;
	volatile int a58 = 5678;
	volatile int a59 = 2345;
	volatile int a60 = 6789;
	volatile int a61 = 3456;
	volatile int a62 = 7890;
	volatile int a63 = 4567;
	volatile int a64 = 1234;
	volatile int a65 = 5678;
	volatile int a66 = 2345;
	volatile int a67 = 6789;
	volatile int a68 = 3456;
	volatile int a69 = 7890;
	volatile int a70 = 4567;
	volatile int a71 = 1234;
	volatile int a72 = 5678;
	volatile int a73 = 2345;
	volatile int a74 = 6789;
	volatile int a75 = 3456;
	volatile int a76 = 7890;
	volatile int a77 = 4567;
	volatile int a78 = 1234;
	volatile int a79 = 5678;
	volatile int a80 = 2345;
	volatile int a81 = 6789;
	volatile int a82 = 3456;
	volatile int a83 = 7890;
	volatile int a84 = 4567;
	volatile int a85 = 1234;
	volatile int a86 = 5678;
	volatile int a87 = 2345;
	volatile int a88 = 6789;
	volatile int a89 = 3456;
	volatile int a90 = 7890;
	volatile int a91 = 4567;
	volatile int a92 = 1234;
	volatile int a93 = 5678;
	volatile int a94 = 2345;
	volatile int a95 = 6789;
	volatile int a96 = 3456;
	volatile int a97 = 7890;
	volatile int a98 = 4567;
	volatile int a99 = 1234;
	volatile int a100 = 5678;
	
	*result = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31 + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39 + a40 + a41 + a42 + a43 + a44 + a45 + a46 + a47 + a48 + a49 + a50 + a51 + a52 + a53 + a54 + a55 + a56 + a57 + a58 + a59 + a60 + a61 + a62 + a63 + a64 + a65 + a66 + a67 + a68 + a69 + a70 + a71 + a72 + a73 + a74 + a75 + a76 + a77 + a78 + a79 + a80 + a81 + a82 + a83 + a84 + a85 + a86 + a87 + a88 + a89 + a90 + a91 + a92 + a93 + a94 + a95 + a96 + a97 + a98 + a99 + a100;

}

int main() {
	// TODO: launch config
	// TODO: synchronization like shared memory?
	//fillRegs<<<BLOCKS_PER_SM * SM, BLOCKS_PER_SM, 1>>>();
	int *result;
	SAFE(cudaHostAlloc(&result, sizeof(int), cudaHostAllocMapped));
	*result = 1;
	fillRegs<<<1, 1>>>(result);
	SAFE(cudaDeviceSynchronize());
	printf("Sum: %d\n", *result);
	return 0;
}

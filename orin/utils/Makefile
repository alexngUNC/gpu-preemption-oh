.PHONY: all clean

FLAGS = -gencode arch=compute_87,code=sm_87
NVCC = /usr/local/cuda-12.6/bin/nvcc

all: constant_cycles_kernel

constant_cycles_kernel: constant_cycles_kernel.cu testbench.h
	$(NVCC) $(FLAGS) constant_cycles_kernel.cu -o bin/constant_cycles_kernel

clean:
	rm -rf bin/*

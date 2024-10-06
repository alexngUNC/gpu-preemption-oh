.PHONY: all clean
FLAGS = -gencode arch=compute_89,code=sm_89 -res-usage -lineinfo
NVCC = /usr/local/cuda-12.6/bin/nvcc

all: constant

constant: constant.cu testbench.h
	$(NVCC) $(FLAGS) constant.cu -o bin/constant

clean:
	rm -f bin/*

.PHONY: all clean

FLAGS = -gencode arch=compute_89,code=sm_89 -res-usage -lineinfo
NVCC = /usr/local/cuda-12.6/bin/nvcc

all: spin

spin: spin.cu testbench.h
	$(NVCC) $(FLAGS) spin.cu -o bin/spin	

clean:
	rm -rf bin/*

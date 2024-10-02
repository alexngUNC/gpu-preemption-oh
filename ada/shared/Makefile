.PHONY: all clean
FLAGS = -gencode arch=compute_89,code=sm_89 -res-usage -lineinfo
NVCC = /usr/local/cuda-12.6/bin/nvcc

all: shared 

shared: shared.cu testbench.h
	$(NVCC) $(FLAGS) shared.cu -o bin/shared

clean:
	rm -f bin/*

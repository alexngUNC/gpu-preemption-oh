.PHONY: all clean
FLAGS = -gencode arch=compute_87,code=sm_87 -res-usage -lineinfo
NVCC = /usr/local/cuda-12.6/bin/nvcc

all: shared no_shared 

shared: shared.cu testbench.h
	$(NVCC) $(FLAGS) shared.cu -o bin/shared

no_shared: no_shared.cu testbench.h
	$(NVCC) $(FLAGS) no_shared.cu -o bin/no_shared

clean:
	rm -f bin/*

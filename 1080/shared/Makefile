.PHONY: all clean
FLAGS = -gencode arch=compute_61,code=sm_61 -res-usage -lineinfo
NVCC = /usr/local/cuda-11.1/bin/nvcc

all: shared carveout_shared static_shared partition_only just_alloc

shared: shared.cu testbench.h
	$(NVCC) $(FLAGS) shared.cu -o bin/shared

carveout_shared: carveout_shared.cu testbench.h
	$(NVCC) $(FLAGS) carveout_shared.cu -o bin/carveout_shared

static_shared: static_shared.cu testbench.h
	$(NVCC) $(FLAGS) static_shared.cu -o bin/static_shared

partition_only: partition_only.cu testbench.h
	$(NVCC) $(FLAGS) partition_only.cu -o bin/partition_only

just_alloc: just_alloc.cu testbench.h
	$(NVCC) $(FLAGS) just_alloc.cu -o bin/just_alloc

clean:
	rm -f bin/*


.PHONY: all clean

DRIVER = -lcuda
FLAGS = -gencode arch=compute_87,code=sm_87
NVCC = /usr/local/cuda-12.6/bin/nvcc

all: constant_cycles_kernel pf_spin

constant_cycles_kernel: constant_cycles_kernel.cu testbench.h
	$(NVCC) $(FLAGS) constant_cycles_kernel.cu -o bin/constant_cycles_kernel

pf_spin: pf_spin.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) $(DRIVER) pf_spin.cu -o bin/pf_spin

clean:
	rm -rf bin/*

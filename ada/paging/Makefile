
.PHONY: all clean

DRIVER = -lcuda
FLAGS = -gencode arch=compute_89,code=sm_89
NVCC = /usr/local/cuda-12.6/bin/nvcc

all: pf_spin

pf_spin: pf_spin.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) $(DRIVER) pf_spin.cu -o bin/pf_spin
clean:
	rm -rf bin/*

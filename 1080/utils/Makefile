.PHONY: all clean

FLAGS = -gencode arch=compute_61,code=sm_61
NVCC ?= nvcc

all: constant_cycles_kernel preemption_logger

constant_cycles_kernel: constant_cycles_kernel.cu testbench.h
	$(NVCC) $(FLAGS) -lcuda constant_cycles_kernel.cu -o bin/constant_cycles_kernel

preemption_logger: task_host_utilities.cu preemption_logger.cu testbench.h
	$(NVCC) $(FLAGS) preemption_logger.cu -o bin/preemption_logger -g

clean:
	rm -rf bin/*

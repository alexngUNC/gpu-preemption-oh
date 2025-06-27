.PHONY: all clean
FLAGS = -gencode arch=compute_61,code=sm_61 -res-usage -lineinfo
#FLAGS = -arch=native -res-usage -lineinfo -maxrregcount 255
NVCC = /usr/local/cuda-11.1/bin/nvcc

all: reg 

reg: reg.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) reg.cu -o bin/reg

clean:
	rm -f bin/*

.PHONY: all clean
FLAGS = -gencode arch=compute_89,code=sm_89 -res-usage -lineinfo
#FLAGS = -arch=native -res-usage -lineinfo -maxrregcount 255
NVCC = /usr/local/cuda-12.6/bin/nvcc

all: reg 

reg: reg.cu testbench.h
	$(NVCC) $(FLAGS) reg.cu -o bin/reg

reg2: reg2.cu testbench.h
	$(NVCC) $(FLAGS) reg2.cu -o bin/reg2

#matmul: matmul.cu testbench.h
#	$(NVCC) $(FLAGS) matmul.cu -o bin/matmul

clean:
	rm -f bin/*

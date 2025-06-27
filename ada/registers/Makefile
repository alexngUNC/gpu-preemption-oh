.PHONY: all clean
FLAGS = -gencode arch=compute_89,code=sm_89 -res-usage -lineinfo
#FLAGS = -arch=native -res-usage -lineinfo -maxrregcount 255
NVCC = /home/jbakita/sabrenet-1tb/usr/local/cuda-12.6/bin/nvcc

all: reg reg2 regSpin

reg: reg.cu testbench.h
	$(NVCC) $(FLAGS) reg.cu -o bin/reg

reg2: reg2.cu testbench.h
	$(NVCC) $(FLAGS) reg2.cu -o bin/reg2

regSpin: regSpin.cu testbench.h
	$(NVCC) $(FLAGS) regSpin.cu -o bin/regSpin

clean:
	rm -f bin/*

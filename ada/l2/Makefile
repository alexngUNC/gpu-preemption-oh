.PHONY: all clean
FLAGS = -gencode arch=compute_89,code=sm_89 -res-usage -lineinfo

L1 = -Xptxas -dlcm=ca 
L2 = -Xptxas -dlcm=cg
WB = -Xptxas -dscm=wb
WT = -Xptxas -dscm=wt

NVCC = /usr/local/cuda-12.6/bin/nvcc

all: l1wb l1wt l2wb l2wt

l1wb: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(WB) l2.cu -o bin/l1wb

l1wt: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(WT) l2.cu -o bin/l1wt

l2wb: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(WB) l2.cu -o bin/l2wb

l2wt: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(WT) l2.cu -o bin/l2wt

clean:
	rm -rf bin/*


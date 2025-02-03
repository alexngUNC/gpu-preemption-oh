.PHONY: all clean
FLAGS = -gencode arch=compute_61,code=sm_61 -res-usage -lineinfo

CG = -Xptxas -dscm=cg
L1 = -Xptxas -dlcm=ca 
L2 = -Xptxas -dlcm=cg
WB = -Xptxas -dscm=wb
WT = -Xptxas -dscm=wt

NVCC ?= nvcc

all: l1wb l1wt l2cg l2wb l2wt

l1wb: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(WB) l2.cu -o bin/l1wb

l1wt: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(WT) l2.cu -o bin/l1wt

l2cg: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(CG) l2.cu -o bin/l2cg

l2wb: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(WB) l2.cu -o bin/l2wb

l2wt: l2.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(WT) l2.cu -o bin/l2wt

clean:
	rm -rf bin/*


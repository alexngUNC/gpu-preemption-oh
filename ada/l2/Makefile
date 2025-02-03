.PHONY: all clean
DRIVER = -lcuda
FLAGS = -gencode arch=compute_89,code=sm_89 -res-usage -lineinfo

CG = -Xptxas -dscm=cg
L1 = -Xptxas -dlcm=ca 
L2 = -Xptxas -dlcm=cg
WB = -Xptxas -dscm=wb
WT = -Xptxas -dscm=wt

NVCC = /usr/local/cuda-12.6/bin/nvcc

all: l1wb l1wt l2cg l2wb l2wt l1ctawb l1ctawt l2ctacg l2ctawb l2ctawt

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

l1ctawb: l2cta.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(WB) $(DRIVER) l2cta.cu -o bin/l1ctawb

l1ctawt: l2cta.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(WT) $(DRIVER) l2cta.cu -o bin/l1ctawt

l2ctacg: l2cta.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(CG) $(DRIVER) l2cta.cu -o bin/l2ctacg

l2ctawb: l2cta.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(WB) $(DRIVER) l2cta.cu -o bin/l2ctawb

l2ctawt: l2cta.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(WT) $(DRIVER) l2cta.cu -o bin/l2ctawt

clean:
	rm -rf bin/*


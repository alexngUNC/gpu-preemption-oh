.PHONY: all clean
FLAGS = -gencode arch=compute_61,code=sm_61 -res-usage -lineinfo

L1 = -Xptxas -dlcm=ca 
L2 = -Xptxas -dlcm=cg
CG = -Xptxas -dscm=cg
WB = -Xptxas -dscm=wb
WT = -Xptxas -dscm=wt

NVCC ?= nvcc

#all: l1cg l1wb l1wt l2cg l2wb l2wt l2HalfSpin l2SpinFillMem
all: walk walkPreempt walkSat

l1cg: l2.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(CG) l2.cu -o bin/l1cg

l1wb: l2.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(WB) l2.cu -o bin/l1wb

l1wt: l2.cu  task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) $(L1) $(WT) l2.cu -o bin/l1wt

l2cg: l2.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(CG) l2.cu -o bin/l2cg

l2wb: l2.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(WB) l2.cu -o bin/l2wb

l2wt: l2.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) $(L2) $(WT) l2.cu -o bin/l2wt

l2HalfSpin: l2HalfSpin.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) l2HalfSpin.cu -o bin/l2HalfSpin

l2SpinFillMem: l2SpinFillMem.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) l2SpinFillMem.cu -o bin/l2SpinFillMem

walk: walk.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) walk.cu -o bin/walk

walkPreempt: walkPreempt.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) walkPreempt.cu -o bin/walkPreempt

walkSat: walkSat.cu task_host_utilities.cu testbench.h
	$(NVCC) $(FLAGS) walkSat.cu -o bin/walkSat

clean:
	rm -rf bin/*


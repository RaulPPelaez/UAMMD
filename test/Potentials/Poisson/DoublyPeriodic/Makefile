
UAMMD_ROOT=../../../../
#You might have to change this if you want to use MKL instead of lapacke. Only one is needed.
LAPACKE_FLAGS=-llapacke -I/usr/include/lapacke
#MKL_FLAGS=-DUSE_MKL -DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64  -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

CUDA_ROOT=/usr/local/cuda

INCLUDEFLAGS= -I$(CUDA_ROOT)/include -I $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party  
BASIC_LINE= $(CUDA_ROOT)/bin/nvcc -O3 -std=c++14 -x cu $(INCLUDEFLAGS) --expt-relaxed-constexpr $(MKL_FLAGS) $(LAPACKE_FLAGS)


all: poisson

poisson: PoissonSlab.cu RepulsivePotential.cuh
	$(BASIC_LINE) PoissonSlab.cu  -o poisson -lcufft
test: all
#	(cd charged_wall; bash test.bash);
	(cd uncharged_wall; bash test.bash);
clean:
	rm -f poisson






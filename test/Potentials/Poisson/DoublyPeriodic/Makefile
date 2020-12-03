
#The target CUDA compute capability(s), if not set it will be autodetected
#ARCH ?= 30 35 50 52 60 61
ARCH=75

#Uncomment to compile in double precision mode, single by default
#DOUBLE_PRECISION=-DDOUBLE_PRECISION

#C++ compiler, I tested up to clang++-5.0
#CXX=clang++-5.0
CXX=g++

#Cuda version (assumed to be in /usr/local/cuda*) You can change this in CUDA_ROOT
#If not set it will be autodetected
CUDA_VER=10.2
CUDA_ROOT=/usr/local/cuda
ifeq ($(CUDA_VER),)
CUDA_VER:=$(shell ls -d /usr/local/cuda*/ | grep -Eo '\-[0-9]+\.[0-9]+' | cut -d- -f2 | sort -grk1 | head -1)
endif

NVCC=nvcc

#The file that will be compiled with all:
FILE=benchmark.cu

LOG_LEVEL=5

#Uncomment to add debug flags to nvcc
#DEBUG= -g -DUAMMD_DEBUG  #-DUSE_NVTX -lnvToolsExt 
#DEBUG=-DUSE_NVTX -lnvToolsExt   -lineinfo -g
#Flags to $(CXX)
CPU= -O3  -march=native -fPIC -Wall -Wextra  -Wno-unused-parameter -Wno-unused-function -Wno-sign-compare  #-fsanitize=address

#If arch was not set, autodetect all GPUs in the system
ifeq ($(ARCH),)
GENCODE_FLAGS:=$(shell printf '\#include<cstdio>\n int main(){int nD;cudaGetDeviceCount(&nD);for(int i=0;i<nD;i++){cudaDeviceProp dp;cudaGetDeviceProperties(&dp, i);std::printf("%%d\\n", dp.major*10+dp.minor);} return 0;}' | $(NVCC) -Wno-deprecated-gpu-targets -x cu - -o /tmp/listarch --run | sort -g -k1 | uniq | awk 'END{system("rm -f /tmp/listarch")}{print "-gencode arch=compute_"$$1",code=sm_"$$1}')
else
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif


UAMMD_ROOT=../../../../
INCLUDEFLAGS= -I$(CUDA_ROOT)/include -I $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party -I/usr/include/lapacke
OPTIONS=$(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DEBUG) 
BASIC_LINE= $(CUDA_ROOT)/bin/nvcc   -O3 -std=c++11 -x cu    $(INCLUDEFLAGS) $(OPTIONS)  -ccbin="$(CXX)" -Xcompiler="$(CPU)" $(GENCODE_FLAGS) -L$(CUDA_ROOT)/lib64 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -src-in-ptx -DUSE_MKL -DMKL_ILP64 -m64 -I${MKLROOT}/include   -L${MKLROOT}/lib/intel64  -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl 

all: poissonSlab poisson

poisson:
	$(BASIC_LINE) Poisson.cu  -o poisson -lcufft -llapack -llapacke -lblas

poissonSlab:
	$(BASIC_LINE) PoissonSlab.cu  -o poisson -lcufft #-llapack -llapacke -lblas

clean:
	rm -f poisson






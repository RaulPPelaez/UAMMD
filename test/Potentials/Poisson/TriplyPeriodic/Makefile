
#The target CUDA compute capability(s), if not set it will be autodetected
#ARCH ?= 30 35 50 52 60 61
ARCH=

#Uncomment to compile in double precision mode, single by default
DOUBLE_PRECISION=-DDOUBLE_PRECISION

#C++ compiler, I tested up to clang++-5.0
#CXX=clang++-5.0
CXX=g++

#Cuda version (assumed to be in /usr/local/cuda*) You can change this in CUDA_ROOT
#If not set it will be autodetected
CUDA_VER=

ifeq ($(CUDA_VER),)
CUDA_VER:=$(shell ls -d /usr/local/cuda*/ | grep -Eo '\-[0-9]+\.[0-9]+' | cut -d- -f2 | sort -grk1 | head -1)
endif
CUDA_ROOT=/usr/local/cuda-$(CUDA_VER)

#The file that will be compiled with all:
FILE=benchmark.cu

LOG_LEVEL=5

#Uncomment to add debug flags to nvcc
#DEBUG=  -src-in-ptx -g -G -DUAMMD_DEBUG

#Flags to $(CXX)
CPU= -O3 -funroll-loops  -fno-math-errno -fno-signed-zeros  -march=native -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-sign-compare
#-ffinite-math-only

#If arch was not set, autodetect all GPUs in the system
ifeq ($(ARCH),)
GENCODE_FLAGS:=$(shell printf '\#include<cstdio>\n int main(){int nD;cudaGetDeviceCount(&nD);for(int i=0;i<nD;i++){cudaDeviceProp dp;cudaGetDeviceProperties(&dp, i);std::printf("%%d\\n", dp.major*10+dp.minor);} return 0;}' | $(CUDA_ROOT)/bin/nvcc -Wno-deprecated-gpu-targets -x cu - -o /tmp/listarch --run | sort -g -k1 | uniq | awk 'END{system("rm -f /tmp/listarch")}{print "-gencode arch=compute_"$$1",code=sm_"$$1}')
else
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif



INCLUDEFLAGS= -I$(CUDA_ROOT)/include -I ../../../../src -I ../../../../src/third_party
OPTIONS=$(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DEBUG)
BASIC_LINE= $(CUDA_ROOT)/bin/nvcc   -O3 -std=c++11 -x cu    $(INCLUDEFLAGS) $(OPTIONS)  -ccbin="$(CXX)" -Xcompiler="$(CPU)" $(GENCODE_FLAGS) -L$(CUDA_ROOT)/lib64 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -lineinfo -src-in-ptx

all: poisson

poisson:
	$(BASIC_LINE) Poisson.cu  -o poisson -lcufft

clean:
	rm -f poisson






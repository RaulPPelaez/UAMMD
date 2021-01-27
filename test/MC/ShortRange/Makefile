#The target CUDA compute capability(s), if set to AUTO it will be autodetected, if empty no arch will be specified
#ARCH ?= 50 52 60 61
#ARCH=
ARCH=AUTO

#Uncomment to compile in double precision mode, single by default
#DOUBLE_PRECISION=-DDOUBLE_PRECISION

#C++ compiler, I tested up to clang++-9.0
#CXX=clang++-9.0
CXX=g++

#Cuda version (assumed to be in /usr/local/cuda*) You can change this in CUDA_ROOT
#If not set it will be autodetected
CUDA_VER=

ifeq ($(CUDA_VER),)
CUDA_VER:=$(shell ls -d /usr/local/cuda*/ | grep -Eo '\-[0-9]+\.[0-9]+' | cut -d- -f2 | sort -grk1 | head -1)
endif
CUDA_ROOT=/usr/local/cuda-$(CUDA_VER)
UAMMD_ROOT=..
NVCC=$(CUDA_ROOT)/bin/nvcc

#The file that will be compiled with "make all":
FILE=benchmark.cu

#This flag controls UAMMD's verbosity, 5 is standard messages, 15 is the maximum debug log level and 0 will only print critical errors. 
LOG_LEVEL=5

#Uncomment to add debug flags to nvcc
#DEBUG=  -src-in-ptx -g -G -DUAMMD_DEBUG

#Flags to $(CXX)
CPU= -O3 -funroll-loops  -fno-math-errno -fno-signed-zeros  -march=native -fPIC #-Wall -Wextra  -Wno-unused-function 

#If arch was not set, autodetect all GPUs in the system
ifeq ($(ARCH),AUTO)
GENCODE_FLAGS:=$(shell printf '#include<cstdio>\n int main(){int nD;cudaGetDeviceCount(&nD);for(int i=0;i<nD;i++){cudaDeviceProp dp;cudaGetDeviceProperties(&dp, i);std::printf("%%d\\n", dp.major*10+dp.minor);} return 0;}' | $(NVCC) -x cu - -o /tmp/listarch --run | sort -g -k1 | uniq | awk 'END{system("rm -f /tmp/listarch")}{print "-gencode arch=compute_"$$1",code=sm_"$$1}')
else
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

UAMMD_ROOT=../../../
INCLUDEFLAGS= -I$(CUDA_ROOT)/include -I $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party
OPTIONS=$(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DEBUG)
BASIC_LINE= $(NVCC) -O3 -std=c++14 $(INCLUDEFLAGS) $(OPTIONS) -ccbin="$(CXX)" -Xcompiler="$(CPU)" $(GENCODE_FLAGS) -L$(CUDA_ROOT)/lib64 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets

all: mc

mc:
	$(BASIC_LINE) MonteCarlo.cu -o mc

clean:
	rm -f mc





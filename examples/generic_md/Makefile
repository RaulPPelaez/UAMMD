#The target CUDA compute capability(s), if set to AUTO it will be autodetected, if empty no arch will be specified
#ARCH ?= 50 52 60 61 70 75
#ARCH=AUTO
ARCH=75
#C++ compiler, I tested up to clang++-9.0
#CXX=clang++-9.0
CXX=g++

UAMMD_ROOT=../../
CUDA_ROOT=/usr/local/cuda
NVCC=$(CUDA_ROOT)/bin/nvcc

#Uncomment to compile in double precision mode, single by default
#DOUBLE_PRECISION=-DDOUBLE_PRECISION

#This flag controls UAMMD's verbosity, 5 is standard messages, 15 is the maximum debug log level and 0 will only print critical errors. 
LOG_LEVEL=5

#Uncomment to add debug flags to nvcc
#DEBUG=  -src-in-ptx -g -G -DUAMMD_DEBUG

#Flags to $(CXX)
CPU= -O3 -march=native -fPIC 

#If arch was not set, autodetect all GPUs in the system
ifeq ($(ARCH),AUTO)
GENCODE_FLAGS:=$(shell printf '#include<cstdio>\n int main(){int nD;cudaGetDeviceCount(&nD);for(int i=0;i<nD;i++){cudaDeviceProp dp;cudaGetDeviceProperties(&dp, i);std::printf("%%d\\n", dp.major*10+dp.minor);} return 0;}' | $(NVCC) -Wno-deprecated-gpu-targets -x cu - -o /tmp/listarch --run | sort -g -k1 | uniq | awk 'END{system("rm -f /tmp/listarch")}{print "-gencode arch=compute_"$$1",code=sm_"$$1}')
else
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

OPTIONS=$(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DEBUG)

INCLUDE_FLAGS= -I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src/ -I$(UAMMD_ROOT)/src/third_party/ -I/usr/include/lapacke -I/usr/include/cblas
LIBRARIES=-lcufft -llapacke -lcublas -lcblas
generic: generic_simulation.cu
	$(NVCC) --expt-relaxed-constexpr -std=c++14 -O3 $(OPTIONS) -ccbin="$(CXX)" -Xcompiler="$(CPU)" $(GENCODE_FLAGS) generic_simulation.cu $(INCLUDE_FLAGS) $(LIBRARIES) -o generic

all: generic

clean:
	rm -f generic

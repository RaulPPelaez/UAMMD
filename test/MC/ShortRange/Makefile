#Uncomment to compile in double precision mode, single by default
#DOUBLE_PRECISION=-DDOUBLE_PRECISION

#C++ compiler, I tested up to clang++-9.0
#CXX=clang++-9.0
CXX=g++

CUDA_ROOT=/usr/local/cuda
UAMMD_ROOT=..
NVCC=$(CUDA_ROOT)/bin/nvcc

#This flag controls UAMMD's verbosity, 5 is standard messages, 15 is the maximum debug log level and 0 will only print critical errors. 
LOG_LEVEL=5

#Uncomment to add debug flags to nvcc
#DEBUG=  -src-in-ptx -g -G -DUAMMD_DEBUG

#Flags to $(CXX)
CPU= -O3 -funroll-loops  -fno-math-errno -fno-signed-zeros  -march=native -fPIC 

UAMMD_ROOT=../../../
INCLUDEFLAGS= -I$(CUDA_ROOT)/include -I $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party
OPTIONS=$(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DEBUG)
BASIC_LINE= $(NVCC) -O3 -std=c++14 $(INCLUDEFLAGS) $(OPTIONS) -ccbin="$(CXX)" -Xcompiler="$(CPU)" -L$(CUDA_ROOT)/lib64 --expt-relaxed-constexpr 

all: mc

mc:
	$(BASIC_LINE) MonteCarlo.cu -o mc

clean:
	rm -f mc





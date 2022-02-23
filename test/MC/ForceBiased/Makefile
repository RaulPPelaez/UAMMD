#Uncomment to compile in double precision mode, single by default
#DOUBLE_PRECISION=-DDOUBLE_PRECISION

#C++ compiler, I tested up to clang++-5.0
#CXX=clang++-5.0
CXX=g++

CUDA_ROOT=/usr/local/cuda

#The file that will be compiled with all:
FILE=ForceBiased.cu

LOG_LEVEL=5

#Uncomment to add debug flags to nvcc
#DEBUG=  -src-in-ptx -g -G -DUAMMD_DEBUG

#Flags to $(CXX)
CPU= -O3 -funroll-loops  -fno-math-errno -fno-signed-zeros  -march=native -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-sign-compare

INCLUDEFLAGS= -I$(CUDA_ROOT)/include -I ../../../src -I ../../../src/third_party
OPTIONS=$(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DEBUG)
BASIC_LINE= $(CUDA_ROOT)/bin/nvcc   -O3 -std=c++14 -x cu    $(INCLUDEFLAGS) $(OPTIONS)  -ccbin="$(CXX)" -Xcompiler="$(CPU)" -L$(CUDA_ROOT)/lib64 --expt-relaxed-constexpr 

all: 
	$(BASIC_LINE)  $(FILE) -o forcebiased

clean:
	rm -f forcebiased





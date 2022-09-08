
LOG_LEVEL=5
CXX=g++
NVCC=nvcc -ccbin=$(CXX)
CUDA_ROOT=$(shell dirname `which nvcc`)/..
UAMMD_ROOT=../../../
#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION
INCLUDEFLAGS=-I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party
LDFLAGS=-lcufft
#DEBUG= -DUAMMD_DEBUG -g -G -src-in-ptx
NVCCFLAGS=-std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION) $(LDFLAGS) $(DEBUG)

all: $(patsubst %.cu, %, $(wildcard *.cu))

%: %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $(@:.out=)

clean: $(patsubst %.cu, %.clean, $(wildcard *.cu))

%.clean:
	rm -f $(@:.clean=)

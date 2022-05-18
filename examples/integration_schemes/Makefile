#Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information 
LOG_LEVEL=5

CXX=g++
NVCC=nvcc -ccbin=$(CXX)
CUDA_ROOT=$(shell dirname `which nvcc`)/..
UAMMD_ROOT=../../
#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION
INCLUDEFLAGS=-I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party -I/usr/include/lapacke -I/usr/include/cblas
LDFLAGS=-lcublas -lcufft -llapacke -lcblas  -lcurand
NVCCFLAGS=-std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION) --extended-lambda --expt-relaxed-constexpr $(LDFLAGS)

all: $(patsubst %.cu, %, $(wildcard *.cu))

%: %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $(@:.out=)

clean: $(patsubst %.cu, %.clean, $(wildcard *.cu))

%.clean:
	rm -f $(@:.clean=)

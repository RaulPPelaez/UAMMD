

NVCC=nvcc
CXX=g++

#Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information
LOG_LEVEL=5

#Uncomment to compile in double precision mode, single by default
DOUBLE_PRECISION=-DDOUBLE_PRECISION

#LAPACKE_FLAGS=-llapacke -I/usr/include/lapacke
MKL_FLAGS=-DUSE_MKL -DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64  -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

UAMMD_ROOT=../../../

INCLUDEFLAGS=-I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party
NVCCFLAGS=-ccbin=$(CXX) -std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION)  $(MKL_FLAGS) $(LAPACKE_FLAGS)

all: dpstokes

dpstokes: DPStokes.cu Makefile
	$(NVCC) $(NVCCFLAGS) -lcufft $< -o $(@:.out=)

clean:
	rm -f dpstokes

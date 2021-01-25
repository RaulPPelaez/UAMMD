
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
GENCODE_FLAGS:=$(shell printf '\#include<cstdio>\n int main(){int nD;cudaGetDeviceCount(&nD);for(int i=0;i<nD;i++){cudaDeviceProp dp;cudaGetDeviceProperties(&dp, i);std::printf("%%d\\n", dp.major*10+dp.minor);} return 0;}' | $(NVCC) -Wno-deprecated-gpu-targets -x cu - -o /tmp/listarch --run | sort -g -k1 | uniq | awk 'END{system("rm -f /tmp/listarch")}{print "-gencode arch=compute_"$$1",code=sm_"$$1}')
else
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif


INCLUDEFLAGS= -I$(CUDA_ROOT)/include -I $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party
OPTIONS=$(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DEBUG)
BASIC_LINE= $(NVCC) -O3 -std=c++14 $(INCLUDEFLAGS) $(OPTIONS) -ccbin="$(CXX)" -Xcompiler="$(CPU)" $(GENCODE_FLAGS) -L$(CUDA_ROOT)/lib64 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets

all: 
	$(BASIC_LINE)  $(FILE)

test: benchmark lj ljMult dpd bdhi bd bonds sph mcnvt fcm dambreak quasi2D temporary_memory neighbour_list_iterator error_handling poisson customPotentials


benchmark:
	$(BASIC_LINE) benchmark.cu -o benchmark

customPotentials:
	$(BASIC_LINE) customPotentials.cu -o custom_potentials

lj:
	$(BASIC_LINE) LJ.cu -o lj

ljMult:
	$(BASIC_LINE) LJMultipleTypes.cu -o ljMult

dpd:
	$(BASIC_LINE) DPD.cu -o dpd

bdhi:
	$(BASIC_LINE) BDHI.cu -llapacke -lcblas -lcurand -lcublas -lcusolver -lcufft  -lgomp -o bdhi

quasi2D:
	$(BASIC_LINE) q2D.cu -lcufft  -o q2D

logo:
	$(BASIC_LINE) logo.cu -lcurand -lcublas -llapacke -lcblas -lcufft  -o logo

fcm:
	$(BASIC_LINE) FCM.cu  -lcufft  -llapacke -lcblas -lcurand -lcublas -lcusolver -o fcm

fib:
	$(BASIC_LINE) FIB.cu -lcurand -lcufft  -o fib

icm:
	$(BASIC_LINE) ICM.cu  -lcufft -lcudart -o icm

lbm:
	$(BASIC_LINE) LBM.cu -o lbm -lpng

bd:
	$(BASIC_LINE) BD.cu -lcurand -o bd

bonds:
	$(BASIC_LINE)  Bonds.cu -lcurand -o bonds

sph:
	$(BASIC_LINE) SPH_test.cu -lcurand -o sph

mcnvt:
	$(BASIC_LINE) MCNVT.cu  -o mcnvt


dambreak:
	$(BASIC_LINE) dambreak.cu  -o dambreak

temporary_memory:
	$(BASIC_LINE) temporary_memory.cu  -o temporary_memory

neighbour_list_iterator:
	$(BASIC_LINE) NeighbourListIterator.cu  -o neighbour_list_iterator

error_handling:
	$(BASIC_LINE) error_handling.cu  -o error_handling

poisson:
	$(BASIC_LINE) Poisson.cu  -o poisson -lcufft

dppoisson:
	$(BASIC_LINE) DoublyPeriodicPoisson.cu  -o dppoisson -lcufft -llapacke

signals:
	$(BASIC_LINE) signals.cu -o signals

parameter_updatable:
	$(BASIC_LINE) ParameterUpdatable.cu -o parameter_updatable

clean:
	rm -f lj dpd nvt bdhi pse icm bd bonds sph a.out ljMult benchmark mcnvt fcm dambreak fib logo q2D temporary_memory neighbour_list_iterator error_handling poisson dppoisson custom_potentials signals parameter_updatable






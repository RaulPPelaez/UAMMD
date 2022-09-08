
NVCC=nvcc
CXX=g++

#Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information
LOG_LEVEL=5

#Uncomment to compile in double precision mode, single by default
DOUBLE_PRECISION=-DDOUBLE_PRECISION

UAMMD_ROOT=../../../

INCLUDEFLAGS=-I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party
NVCCFLAGS=-ccbin=$(CXX) -std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION)

all: fcm

fcm: FCM.cu Makefile
	$(NVCC) $(NVCCFLAGS) -lcufft $< -o $(@:.out=)

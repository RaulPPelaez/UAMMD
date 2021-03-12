#Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information 
LOG_LEVEL=5

NVCC=nvcc
CUDA_ROOT=/usr/local/cuda
#Python is required for the python_wrappers example
PYTHON=python3
UAMMD_ROOT=../../
#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION
INCLUDEFLAGS=-I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party
NVCCFLAGS=-std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION) --extended-lambda --expt-relaxed-constexpr

PYTHON_MODULE_NAME=uammd
PYTHON_LIBRARY_NAME=$(PYTHON_MODULE_NAME)$(shell $(PYTHON)-config --extension-suffix)
all: $(patsubst %.cu, %, $(filter-out python_wrapper.cu, $(wildcard *.cu))) $(PYTHON_LIBRARY_NAME)

%: %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(PYTHON_LIBRARY_NAME): python_wrapper.cu Makefile pybind11
	$(NVCC)  $(NVCCFLAGS) $< -o $(PYTHON_LIBRARY_NAME) -w -shared -I pybind11/include -Xcompiler "-fPIC -w" `$(PYTHON)-config --includes` 

pybind11:
	git clone https://github.com/pybind/pybind11

clean: $(patsubst %.cu, %.clean, $(wildcard *.cu))
	rm -rf pybind11
	rm -f $(PYTHON_LIBRARY_NAME)

%.clean:
	rm -f $(@:.clean=)

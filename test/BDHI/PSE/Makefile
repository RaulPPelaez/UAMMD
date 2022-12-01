#Uncomment to compile in double precision mode, single by default
#DOUBLE_PRECISION=-DDOUBLE_PRECISION

#C++ compiler, I tested up to clang++-5.0
#CXX=clang++-5.0
CXX=g++

LOG_LEVEL=5

#Uncomment to add debug flags to nvcc
#DEBUG= -lineinfo -src-in-ptx -g -G

#Flags to $(CXX)
CPU= -O3 -funroll-loops -ffinite-math-only -fno-math-errno -fno-signed-zeros -march=native -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-sign-compare

UAMMD_ROOT=../../../
INCLUDEFLAGS= -I $(UAMMD_ROOT)/src -I $(UAMMD_ROOT)/src/third_party
OPTIONS=$(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DEBUG)
BASIC_LINE= nvcc   -O3 -std=c++14 -x cu    $(INCLUDEFLAGS) $(OPTIONS)  -ccbin="$(CXX)" -Xcompiler="$(CPU)"  --expt-relaxed-constexpr

all: pse

pse:
	$(BASIC_LINE) PSE.cu  -lcurand -lcublas -lcufft  -o pse

clean:
	rm -f pse


#WARNING: You should probably change the compute architecture for your GPU in BASIC_LINE or here in ARCH
#The target CUDA compute capability	
ARCH=52

#Change for your system, HG needs to be already compiled
HYDROGRID_SRC=/home/raul/Dropbox/Trabajo/HydroGrid/src/

#Uncomment to compile in double precision mode

#DOUBLE_PRECISION=-DDOUBLE_PRECISION

UAMMD_SRC=../../src

CPU= -O3 -funroll-loops -ffinite-math-only -fno-signaling-nans -fno-math-errno -fno-signed-zeros -frename-registers -march=native -fPIC

DEBUG= -O3

CXX=g++
BASIC_LINE= nvcc  $(DOUBLE_PRECISION) -lineinfo -I  $(UAMMD_SRC) -I $(UAMMD_SRC)/third_party  -O3 -ccbin="$(CXX)" -Xcompiler="$(CPU)"  -src-in-ptx -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -x cu -std=c++11 --expt-relaxed-constexpr
all: hg

hg:
	$(BASIC_LINE) -I$(HYDROGRID_SRC) hydroGridAnalysis.cu  -L$(HYDROGRID_SRC) -lCallHydroGrid
	cp $(HYDROGRID_SRC)/libCallHydroGrid.so .

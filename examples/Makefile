
#WARNING: You should probably change the compute architecture for your GPU in BASIC_LINE

FILE=benchmark.cu

CPU= -O3 -funroll-loops -ffinite-math-only -fno-signaling-nans -fno-math-errno -fno-signed-zeros -frename-registers -march=native -fPIC

DEBUG= -O3
#DOUBLE_PRECISION=-DDOUBLE_PRECISION

CXX=g++
BASIC_LINE= nvcc  $(DOUBLE_PRECISION) -lineinfo -I  ../src -I ../src/third_party  -O3 -ccbin="$(CXX)" -Xcompiler="$(CPU)"  -src-in-ptx -gencode arch=compute_52,code=sm_52 -x cu -std=c++11 --expt-relaxed-constexpr
all:
	$(BASIC_LINE)  $(FILE)

lj:
	$(BASIC_LINE) LJ.cu

dpd:
	$(BASIC_LINE) DPD.cu

nvt:
	$(BASIC_LINE) nvt.cu

bdhi:
	$(BASIC_LINE) BDHI.cu -llapacke -lcblas -lcurand -lcublas -lcusolver -lcufft -lnvToolsExt

pse:
	$(BASIC_LINE) PSE.cu -lcurand -lcublas -lcusolver -llapacke -lcblas -lcufft -lcudart -lnvToolsExt

bd:
	$(BASIC_LINE) BD.cu -lcurand

bonds:
	$(BASIC_LINE)  Bonds.cu -lcurand 

sph:
	$(BASIC_LINE) SPH_test.cu -lcurand

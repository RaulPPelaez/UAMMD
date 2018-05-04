
#WARNING: You should probably change the compute architecture for your GPU in BASIC_LINE or here in ARCH
#The target CUDA compute capability	
ARCH=52

FILE=benchmark.cu

CPU= -O3 -funroll-loops -ffinite-math-only -fno-signaling-nans -fno-math-errno -fno-signed-zeros -frename-registers -march=native -fPIC -Wall -Wextra -Wno-unused-parameter

DEBUG= -O3
LOG_LEVEL=6

#DOUBLE_PRECISION=-DDOUBLE_PRECISION

CXX=g++
BASIC_LINE= nvcc  $(DOUBLE_PRECISION) -DMAXLOGLEVEL=$(LOG_LEVEL) -lineinfo -I  ../src -I ../src/third_party  -O3 -ccbin="$(CXX)" -Xcompiler="$(CPU)"  -src-in-ptx -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -x cu -std=c++11 --expt-relaxed-constexpr
all:
	$(BASIC_LINE)  $(FILE)

test: lj dpd nvt bdhi pse bd bonds sph

lj:
	$(BASIC_LINE) LJ.cu -o lj

dpd:
	$(BASIC_LINE) DPD.cu -o dpd

nvt:
	$(BASIC_LINE) nvt.cu -o nvt

bdhi:
	$(BASIC_LINE) BDHI.cu -llapacke -lcblas -lcurand -lcublas -lcusolver -lcufft -lnvToolsExt -o bdhi

pse:
	$(BASIC_LINE) PSE.cu -lcurand -lcublas -lcusolver -llapacke -lcblas -lcufft -lcudart -lnvToolsExt -o pse

icm:
	$(BASIC_LINE) ICM.cu  -lcufft -lcudart -o icm

bd:
	$(BASIC_LINE) BD.cu -lcurand -o bd

bonds:
	$(BASIC_LINE)  Bonds.cu -lcurand -o bonds

sph:
	$(BASIC_LINE) SPH_test.cu -lcurand -o sph

clean:
	rm -f lj dpd nvt bdhi pse icm bd bonds sph a.out






FILE=LJ.cu

CPU= -O3 -funroll-loops -ffinite-math-only -fno-signaling-nans -fno-math-errno -fno-signed-zeros -frename-registers -march=native -fPIC

DEBUG= -O3
all:
	nvcc  -lineinfo -I ../src -O3 -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  $(FILE) --expt-relaxed-constexpr  -lnvToolsExt -lcurand

bdhi:
	nvcc  -lineinfo -I ../src -O3 -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  BDHI.cu --expt-relaxed-constexpr  -lnvToolsExt -lcurand -lcublas -lcusolver -llapacke -lcblas


pse:
	nvcc  -lineinfo -I ../src -O3 -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  PSE.cu --expt-relaxed-constexpr  -lnvToolsExt -lcurand -lcublas -lcusolver -llapacke -lcblas -lcufft -lcudart

bd:
	nvcc  -lineinfo -I ../src -O3 -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  BD.cu --expt-relaxed-constexpr  -lnvToolsExt -lcurand

bonds:
	nvcc   -lineinfo -I ../src $(DEBUG) -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  Bonds.cu --expt-relaxed-constexpr  -lnvToolsExt -lcurand 

sph:
	nvcc   -lineinfo -I ../src $(DEBUG) -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  SPH_test.cu --expt-relaxed-constexpr  -lnvToolsExt -lcurand

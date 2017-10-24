
CPU= -O3 -funroll-loops -m64 -ffinite-math-only -fno-signaling-nans -fno-math-errno -fno-signed-zeros -frename-registers -march=native -fPIC

all:
	nvcc  -lineinfo -I ../src -O3 -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  LJ.cu --expt-relaxed-constexpr  -lnvToolsExt -lcurand

bdhi:
	nvcc  -lineinfo -I ../src -O3 -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  BDHI.cu --expt-relaxed-constexpr  -lnvToolsExt -lcurand -lcublas -lcusolver -llapacke -lcblas


pse:
	nvcc  -lineinfo -I ../src -O3 -ccbin="g++" -Xcompiler="$(CPU)"  -src-in-ptx -arch=sm_52 -x cu -std=c++11  PSE.cu --expt-relaxed-constexpr  -lnvToolsExt -lcurand -lcublas -lcusolver -llapacke -lcblas -lcufft -lcudart

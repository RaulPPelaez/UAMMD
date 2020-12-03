/*Raul P. Pelaez 2020. Fast Chebyshev Transform for the Doubly Periodic Poisson solver. Slab geometry
*/

#ifndef DPPOISSONSLAB_FASTCHEBYSHEVTRANSFORM_CUH
#define DPPOISSONSLAB_FASTCHEBYSHEVTRANSFORM_CUH

#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include "System/System.h"
#include "global/defines.h"
#include "misc/ChevyshevUtils.cuh"
#include"utils/cufftDebug.h"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftComplex4.cuh"
#include "utils/cufftComplex2.cuh"
namespace uammd{
  namespace DPPoissonSlab_ns{

    namespace fct_ns{

      template<class T>
      __global__ void periodicExtension(T* signal, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= n.x*n.y*n.z) return;
	const int ikx = id%n.x;
	const int iky = (id/n.x)%n.y;
	const int iz  = id/(n.x*n.y);
	if(iz>=n.z-1 or iz == 0) return;
	const int zf = 2*n.z-2-iz;
	const int zi = iz;
	signal[ikx+n.x*iky+n.x*n.y*zf]  = signal[ikx+n.x*iky+n.x*n.y*zi];
      }

      template<class T>
      __global__ void scaleFFTToForwardChebyshevTransform(T* signal, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= n.x*n.y*n.z) return;
	const int iz  = id/(n.x*n.y);
	signal[id] *= real(1.0)/real(n.z-1);
	if(iz==0 or iz == n.z-1){
	  signal[id] *= real(0.5);
	}
      }

      template<class T>
      __global__ void scaleFFTToInverseChebyshevTransform(T* signal, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= (n.x/2+1)*n.y*n.z) return;
	const int iz  = id/((n.x/2+1)*n.y);
	signal[id] *= real(0.5)/real(n.x*n.y);
	if(iz==0 or iz == n.z-1){
	  signal[id] *= real(2.0);
	}
      }

      __global__ void transf(cufftComplex4* in, cufftComplex* out, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int nel = (n.x/2+1)*n.y*(2*n.z-2);
	if(id>= nel) return;
	cufftComplex4 Ei = in[id];
	out[id] = Ei.x;
	out[nel+id] = Ei.y;
	out[2*nel+id] = Ei.z;
	out[3*nel+id] = Ei.w;
      }

      __global__ void transf2(real* in, real4* out, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int nel = 2*(n.x/2+1)*n.y*(2*n.z-2);
	if(id>= nel) return;
	real Ex = in[id];
	real Ey = in[nel + id];
	real Ez = in[2*nel + id];
	real P = in[3*nel + id];
	out[id] = make_real4(Ex, Ey, Ez, P);
      }

    }

    class FastChebyshevTransform{
      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      gpu_container<char> cufftWorkArea;
      std::shared_ptr<System> sys;
      int3 gridSize;
    public:

      FastChebyshevTransform(std::shared_ptr<System> sys, int3 gs):
	sys(sys), gridSize(gs){
	sys->log<System::DEBUG>("[SpreadInterpolate] Initialized");
	initCuFFT();
      }

      template<class RealContainer>
      cached_vector<cufftComplex> forwardTransform(RealContainer &gridData, cudaStream_t st){
	sys->log<System::DEBUG2>("[DPPoissonSlab] Taking grid charges to wave/Chebyshev space");
	CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
	const int3 n = gridSize;
	real* d_gridData = (real*)thrust::raw_pointer_cast(gridData.data());
	cached_vector<cufftComplex> gridDataFourier((2*n.z-2)*n.y*(n.x/2+1));
        cufftComplex* d_gridDataFourier = thrust::raw_pointer_cast(gridDataFourier.data());
	const int blockSize = 128;
	int nblocks = ((2*(n.x/2+1))*n.y*n.z)/blockSize+1;
	CudaCheckError();
        fct_ns::periodicExtension<<<nblocks, blockSize, 0, st>>>(d_gridData, make_int3(2*(n.x/2+1), n.y, n.z));
	CudaCheckError();
	CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, d_gridData, d_gridDataFourier));
	nblocks = ((n.x/2+1)*n.y*n.z)/blockSize+1;
	CudaCheckError();
	fct_ns::scaleFFTToForwardChebyshevTransform<<<nblocks, blockSize, 0, st>>>(d_gridDataFourier, make_int3((n.x/2+1), n.y, n.z));
	CudaCheckError();
	return gridDataFourier;
      }

      template<class Cufft4Container>
      cached_vector<real4> inverseTransform(Cufft4Container & gridDataFourier, cudaStream_t st){
	sys->log<System::DEBUG2>("[DPPoissonSlab] Force and energy to real space");
	CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));
	const int3 n = gridSize;
	cufftComplex4* d_gridDataFourier = thrust::raw_pointer_cast(gridDataFourier.data());
	const int blockSize = 128;
	const int nblocks = ((n.x/2+1)*n.y*n.z)/blockSize+1;
	fct_ns::scaleFFTToInverseChebyshevTransform<<<nblocks, blockSize, 0, st>>>(d_gridDataFourier, make_int3(n.x, n.y, n.z));
	fct_ns::periodicExtension<<<nblocks, blockSize, 0, st>>>(d_gridDataFourier, make_int3((n.x/2+1), n.y, n.z));
	//CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, (cufftComplex*)d_gridDataFourier, (cufftReal*)d_gridData));

	cached_vector<real> gridDataR(4*(2*n.z-2)*n.y*2*(n.x/2+1));
	real* d_gridDataR = thrust::raw_pointer_cast(gridDataR.data());
	{
	  cached_vector<cufftComplex> gridDataFouR(4*(2*n.z-2)*n.y*(n.x/2+1));
	  cufftComplex* d_gridDataFouR = thrust::raw_pointer_cast(gridDataFouR.data());
	  fct_ns::transf<<<((n.x/2+1)*n.y*(2*n.z-2))/blockSize+1, blockSize, 0, st>>>(d_gridDataFourier, d_gridDataFouR, n);
	  CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, d_gridDataFouR, d_gridDataR));
	}
	cached_vector<real4> gridData((2*n.z-2)*n.y*2*(n.x/2+1));
	real4* d_gridData = thrust::raw_pointer_cast(gridData.data());
	fct_ns::transf2<<<(2*(n.x/2+1)*n.y*(2*n.z-2))/blockSize+1, blockSize, 0, st>>>(d_gridDataR, d_gridData, n);
	CudaCheckError();
	return gridData;
      }


    private:

      void initCuFFT(){
	sys->log<System::DEBUG>("[DPPoissonSlab] Initialize cuFFT");
	CufftSafeCall(cufftCreate(&cufft_plan_forward));
	CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
	size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
	//This sizes have to be reversed according to the cufft docs
	int3 n = gridSize;
	int3 cdtmp = {2*(n.z-1), n.y, n.x};
	int3 inembed = {2*(n.z-1), n.y, 2*(n.x/2+1)};
	int3 oembed = {2*(n.z-1), n.y, n.x/2+1};
	CufftSafeCall(cufftMakePlanMany(cufft_plan_forward,
					3, &cdtmp.x,
					&inembed.x,
					1, 1,
					&oembed.x,
					1, 1,
					CUFFT_Real2Complex<real>::value, 1,
					&cufftWorkSizei));
	CufftSafeCall(cufftCreate(&cufft_plan_inverse));
	CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));
	CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
					3, &cdtmp.x,
					&oembed.x,
					1, oembed.x*oembed.y*oembed.z,
					&inembed.x,
					1, inembed.x*inembed.y*inembed.z,
					CUFFT_Complex2Real<real>::value, 4,
					&cufftWorkSizei));
	sys->log<System::DEBUG>("[DPPoissonSlab] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
	size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei);
	try{
	  cufftWorkArea.resize(cufftWorkSize);
	}
	catch(thrust::system_error &e){
	  System::log<System::EXCEPTION>("[FastChebyshevTransform] Could not allocate cuFFT memory");
	  throw;
	}
	auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
	CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
	CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
	CudaCheckError();
      }

    };

  }
}


#endif

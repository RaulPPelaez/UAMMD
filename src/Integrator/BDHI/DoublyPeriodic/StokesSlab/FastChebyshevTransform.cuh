/*Raul P. Pelaez 2020-2021. Fast Chebyshev Transform for the Doubly Periodic Stokes solver.
*/

#ifndef DPSTOKESSLAB_FASTCHEBYSHEVTRANSFORM_CUH
#define DPSTOKESSLAB_FASTCHEBYSHEVTRANSFORM_CUH

#include "utils.cuh"
#include "System/System.h"
#include "global/defines.h"
#include "misc/ChevyshevUtils.cuh"
#include"utils/cufftDebug.h"
#include "utils/cufftPrecisionAgnostic.h"
namespace uammd{
  namespace DPStokesSlab_ns{

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
      __global__ void scaleFFTToForwardChebyshevTransform(T* signalin, T*signalout, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= n.x*n.y*n.z) return;
	const int iz  = id/(n.x*n.y);
	const int ikx = id%n.x;
	const int iky = (id/n.x)%n.y;
	if(iz>0 and iz < (n.z-1))
	  signalout[id] = signalin[id] + signalin[ikx+n.x*(iky+n.y*(2*n.z-2-iz))];
	signalout[id] *= real(0.5)/real(n.z-1);
      }

      template<class T>
      __global__ void scaleFFTToInverseChebyshevTransform(T* signal, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= (n.x/2+1)*n.y*n.z) return;
	const int iz  = id/((n.x/2+1)*n.y);
	signal[id] *= ((iz==0 or iz == n.z-1)?real(1.0):real(0.5))/real(n.x*n.y*(2*n.z-2));
	signal[id] *= real(2.0)*n.z-real(2.0);
      }

      template<class Tin, class Tout>
      __global__ void unpack(Tin* in, Tout* out, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int nel = n.x*n.y*n.z;
	if(id>= nel) return;
        auto Ei = in[id];
	out[id] = Ei.x;
	out[nel+id] = Ei.y;
	out[2*nel+id] = Ei.z;
	out[3*nel+id] = Ei.w;
      }
      
      template<class Tin, class Tout>
      __global__ void pack(Tin* in, Tout* out, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	int nel = n.x*n.y*n.z;
	if(id>= nel) return;
	auto Ex = in[id];
	auto Ey = in[nel + id];
	auto Ez = in[2*nel + id];
	auto P = in[3*nel + id];
	out[id] = {Ex, Ey, Ez, P};
      }

    }

    class FastChebyshevTransform{
      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      gpu_container<char> cufftWorkArea;
      int3 gridSize;
    public:

      FastChebyshevTransform(int3 gs):
	gridSize(gs){
	System::log<System::DEBUG>("[FastChebyshevTransform] Initialized");
	initCuFFT();
      }
      //In order to use cufft correctly I have to go back and forth between a AoS and SoA layout, there has to be a better way. 
      template<class Real4Container>
      cached_vector<cufftComplex4> forwardTransform(Real4Container &gridData, cudaStream_t st){
	System::log<System::DEBUG2>("[DPStokesSlab] Taking forces to wave/Chebyshev space");
	CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
	const int3 n = gridSize;
	real4* d_gridData = (real4*)thrust::raw_pointer_cast(gridData.data());
	const int blockSize = 128;
	int nblocks = ((2*(n.x/2+1))*n.y*n.z)/blockSize+1;
	CudaCheckError();
        fct_ns::periodicExtension<<<nblocks, blockSize, 0, st>>>(d_gridData, make_int3(2*(n.x/2+1), n.y, n.z));	
	CudaCheckError();
	cached_vector<cufftComplex> gridDataFouR(4*(2*n.z-2)*n.y*(n.x/2+1));
	cufftComplex* d_gridDataFouR = thrust::raw_pointer_cast(gridDataFouR.data());
	{
	  cached_vector<real> gridDataR(4*(2*n.z-2)*n.y*2*(n.x/2+1));
	  real* d_gridDataR = thrust::raw_pointer_cast(gridDataR.data());
	  fct_ns::unpack<<<(2*(n.x/2+1)*n.y*(2*n.z-2))/blockSize+1, blockSize, 0, st>>>(d_gridData, d_gridDataR,
											make_int3(2*(n.x/2+1), n.y, 2*n.z-2));
	  CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, d_gridDataR, d_gridDataFouR));
	  CudaCheckError();
	}
	cached_vector<cufftComplex4> gridDataFourier(2*(2*n.z-2)*n.y*(n.x/2+1));
	System::log<System::DEBUG2>("[DPStokesSlab] Taking forces to wave/Chebyshev space");
        cufftComplex4* d_gridDataFourier = thrust::raw_pointer_cast(gridDataFourier.data());
	fct_ns::pack<<<((n.x/2+1)*n.y*(2*n.z-2))/blockSize+1, blockSize, 0, st>>>(d_gridDataFouR, d_gridDataFourier,
										  make_int3(n.x/2+1, n.y, 2*n.z-2));
	auto gridDataFourier2 = gridDataFourier;
	cufftComplex4* d_gridDataFourier2 = thrust::raw_pointer_cast(gridDataFourier2.data());
	CudaCheckError();
	nblocks = ((n.x/2+1)*n.y*n.z)/blockSize+1;
	fct_ns::scaleFFTToForwardChebyshevTransform<<<nblocks, blockSize, 0, st>>>(d_gridDataFourier2, d_gridDataFourier, make_int3((n.x/2+1), n.y, n.z));
	cudaDeviceSynchronize();
	CudaCheckError();
	System::log<System::DEBUG2>("[DPStokesSlab] Taking forces to wave/Chebyshev space");
	return gridDataFourier;
      }

      template<class Cufft4Container>
      cached_vector<real4> inverseTransform(Cufft4Container & gridDataFourier, cudaStream_t st){
	System::log<System::DEBUG2>("[DPStokesSlab] Velocity and pressure to real space");
	CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));
	const int3 n = gridSize;
	cufftComplex4* d_gridDataFourier = thrust::raw_pointer_cast(gridDataFourier.data());
	const int blockSize = 128;
	const int nblocks = ((n.x/2+1)*n.y*n.z)/blockSize+1;
	fct_ns::scaleFFTToInverseChebyshevTransform<<<nblocks, blockSize, 0, st>>>(d_gridDataFourier, make_int3(n.x, n.y, n.z));
	fct_ns::periodicExtension<<<nblocks, blockSize, 0, st>>>(d_gridDataFourier, make_int3((n.x/2+1), n.y, n.z));
	cached_vector<real> gridDataR(4*(2*n.z-2)*n.y*2*(n.x/2+1));
	real* d_gridDataR = thrust::raw_pointer_cast(gridDataR.data());
	{
	  cached_vector<cufftComplex> gridDataFouR(4*(2*n.z-2)*n.y*(n.x/2+1));
	  cufftComplex* d_gridDataFouR = thrust::raw_pointer_cast(gridDataFouR.data());
	  fct_ns::unpack<<<((n.x/2+1)*n.y*(2*n.z-2))/blockSize+1, blockSize, 0, st>>>(d_gridDataFourier, d_gridDataFouR,
										      make_int3(n.x/2+1, n.y, 2*n.z-2));
 
	  CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, d_gridDataFouR, d_gridDataR));
	}
	cached_vector<real4> gridData((2*n.z-2)*n.y*2*(n.x/2+1));
	real4* d_gridData = thrust::raw_pointer_cast(gridData.data());
	fct_ns::pack<<<(2*(n.x/2+1)*n.y*(2*n.z-2))/blockSize+1, blockSize, 0, st>>>(d_gridDataR, d_gridData,
										    make_int3(2*(n.x/2+1), n.y, 2*n.z-2));
	CudaCheckError();
	return gridData;
      }


    private:

      void initCuFFT(){
	System::log<System::DEBUG>("[DPStokesSlab] Initialize cuFFT");
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
					1, inembed.x*inembed.y*inembed.z,
					&oembed.x,
					1, oembed.x*oembed.y*oembed.z,
					CUFFT_Real2Complex<real>::value, 3,
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
	System::log<System::DEBUG>("[DPStokesSlab] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
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

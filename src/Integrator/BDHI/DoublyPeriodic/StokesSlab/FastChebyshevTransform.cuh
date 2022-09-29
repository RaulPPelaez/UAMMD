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
#include"utils/NVTXTools.h"
namespace uammd{
  namespace DPStokesSlab_ns{

    namespace fct_ns{

      template<class T>
      __global__ void periodicExtension(DataXYZPtr<T> signal, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= n.x*n.y*n.z) return;
	const int ikx = id%n.x;
	const int iky = (id/n.x)%n.y;
	const int iz  = id/(n.x*n.y) + n.z;
	//	if(iz>=n.z-1 or iz == 0) return;
	if(iz>=2*n.z-2) return;
	int src = ikx+(iky+(2*n.z-2-iz)*n.y)*n.x;
	int dest = ikx+(iky+iz*n.y)*n.x;
		// signal[ikx+n.x*iky+n.x*n.y*zf]  = signal[ikx+n.x*iky+n.x*n.y*zi];
	signal.x()[dest] = signal.x()[src];
	signal.y()[dest] = signal.y()[src];
	signal.z()[dest] = signal.z()[src];
      }

      template<class T>
      __global__ void scaleFFTToForwardChebyshevTransform(const DataXYZPtr<T> signalin,
							  DataXYZPtr<T> signalout, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= n.x*n.y*n.z) return;
	const int iz  = id/(n.x*n.y);
	const int ikx = id%n.x;
	const int iky = (id/n.x)%n.y;
	int dest = ikx+(iky+iz*n.y)*n.x;
	auto pm = (iz==0 or iz==(n.z-1))?real(1.0):real(2.0);
	signalout.x()[dest] = pm/(2*n.z-2)*signalin.x()[dest];
	signalout.y()[dest] = pm/(2*n.z-2)*signalin.y()[dest];
	signalout.z()[dest] = pm/(2*n.z-2)*signalin.z()[dest];


	// if(iz>0 and iz < (n.z-1))
	//   signalout[id] = signalin[id] + signalin[ikx+n.x*(iky+n.y*(2*n.z-2-iz))];
	// signalout[id] *= real(0.5)/real(n.z-1);
	      }

      template<class T>
      __global__ void scaleFFTToInverseChebyshevTransform(DataXYZPtr<T> signal, int3 n){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= (n.x/2+1)*n.y*n.z) return;
	const int iz  = id/((n.x/2+1)*n.y);
	// signal[id] *= ((iz==0 or iz == n.z-1)?real(1.0):real(0.5))/real(n.x*n.y*(2*n.z-2));
	// signal[id] *= real(2.0)*n.z-real(2.0);
	auto pm = (iz==0 or iz==(n.z-1))?real(1.0):real(2.0);
	signal.x()[id] *= real(1.0)/(pm*n.x*n.y);
	signal.y()[id] *= real(1.0)/(pm*n.x*n.y);
	signal.z()[id] *= real(1.0)/(pm*n.x*n.y);

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
      auto forwardTransform(DataXYZ<real> &gridData, cudaStream_t st){
	System::log<System::DEBUG2>("[DPStokesSlab] Transforming to wave/Chebyshev space");
	CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
	const int3 n = gridSize;
	const int blockSize = 128;
	int nblocks = ((2*(n.x/2+1))*n.y*n.z)/blockSize+1;
	CudaCheckError();
        fct_ns::periodicExtension<<<nblocks, blockSize, 0, st>>>(DataXYZPtr<real>(gridData),
								 make_int3(2*(n.x/2+1), n.y, n.z));
	CudaCheckError();
	DataXYZ<complex> gridDataFou((2*n.z-2)*n.y*(n.x/2+1));
	CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, gridData.x(), gridDataFou.x()));
	CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, gridData.y(), gridDataFou.y()));
	CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, gridData.z(), gridDataFou.z()));
	nblocks = ((n.x/2+1)*n.y*n.z)/blockSize+1;
	auto d_gridDataFou = DataXYZPtr<complex>(gridDataFou);
	fct_ns::scaleFFTToForwardChebyshevTransform<<<nblocks, blockSize, 0, st>>>
	  (d_gridDataFou, d_gridDataFou, make_int3((n.x/2+1), n.y, n.z));
	cudaDeviceSynchronize();
	CudaCheckError();
	System::log<System::DEBUG2>("[DPStokesSlab] Taking forces to wave/Chebyshev space");
	return gridDataFou;
      }

      auto inverseTransform(DataXYZ<complex> & gridDataFourier, cudaStream_t st){
	System::log<System::DEBUG2>("[DPStokesSlab] Transforming to real space");
	CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));
	const int3 n = gridSize;
	const int blockSize = 128;
	const int nblocks = ((n.x/2+1)*n.y*n.z)/blockSize+1;
	auto d_gridDataFou = DataXYZPtr<complex>(gridDataFourier);
	fct_ns::scaleFFTToInverseChebyshevTransform<<<nblocks, blockSize, 0, st>>>(d_gridDataFou,
										   make_int3(n.x, n.y, n.z));
	fct_ns::periodicExtension<<<nblocks, blockSize, 0, st>>>(d_gridDataFou, make_int3((n.x/2+1), n.y, n.z));
	DataXYZ<real> gridData(2*(n.x/2+1)*n.y*(2*n.z-2));
	CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, gridDataFourier.x(), gridData.x()));
	CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, gridDataFourier.y(), gridData.y()));
	CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, gridDataFourier.z(), gridData.z()));
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
					CUFFT_Real2Complex<real>::value, 1,
					&cufftWorkSizef));
	CufftSafeCall(cufftCreate(&cufft_plan_inverse));
	CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));
	CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
					3, &cdtmp.x,
					&oembed.x,
					1, oembed.x*oembed.y*oembed.z,
					&inembed.x,
					1, inembed.x*inembed.y*inembed.z,
					CUFFT_Complex2Real<real>::value, 1,
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

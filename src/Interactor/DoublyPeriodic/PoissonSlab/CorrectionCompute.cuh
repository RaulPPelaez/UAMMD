/*Raul P. Pelaez 2020. Potential and field correction for the Doubly Periodic Poisson solver. Slab geometry
*/

#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftDebug.h"
#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/MismatchCompute.cuh"
#include<fstream>
#include<iomanip>
namespace uammd{
  namespace DPPoissonSlab_ns{

    __device__ cufftComplex2 analyticalCorrectionNonMetallic(double k, double z, double H, Permitivity perm, MismatchVals mis){
      const double eb = perm.bottom/perm.inside;
      const double et = perm.top/perm.inside;
      const double denominator = (exp(double(-2.0)*H*k)*(double(-1.0) + eb + et - eb*et) + (double(1.0) + eb)*(double(1.0) + et));
      //This expression is written so the highest exponential evaluated is exp(kmax*HE)
      auto ezm2H = exp(k*(z-double(2.0)*H))*(double(-1.0) + et)*(-mis.mE0/perm.inside - eb*k*mis.mP0);
      auto emz = exp(-z*k)*(double(1.0) + et)*(-mis.mE0/perm.inside - eb*k*mis.mP0);
      auto emHmz = exp((-H-z)*k)*(double(-1.0) + eb)* (-mis.mEH/perm.inside + et*k*mis.mPH);
      auto emHpz = exp(k*(-H + z))*(double(1.0) + eb)*(-mis.mEH/perm.inside + et*k*mis.mPH);
      cufftComplex2 EzAndPhi;
      //field Z
      EzAndPhi.x = (ezm2H + emz + emHmz + emHpz) / denominator;
      //phi
      EzAndPhi.y = (-ezm2H + emz + emHmz - emHpz) / (denominator * k);
      return EzAndPhi;
    }
    
    __device__ cufftComplex2 analyticalCorrectionOneMetallicWall(double k, double z, double H, Permitivity perm, MismatchVals mis, bool isMetallicTop){
      if(isMetallicTop){
	const double eb = perm.bottom/perm.inside;      
	const double denominator = k*(exp(-k*H)*(1.0-eb) + exp(k*H)*(1.0+eb));
	const cufftComplex Ai = (-(1.0+eb)*k*mis.mPH + exp(-k*H)*(mis.mE0/perm.inside + eb*k*mis.mP0))/denominator;
	const cufftComplex Bi = ((-1.0+eb)*k*mis.mPH + exp(k*H)*(-mis.mE0/perm.inside - eb*k*mis.mP0))/denominator;
	cufftComplex2 EzAndPhi;
	//field Z
	EzAndPhi.x = -(k*Ai*exp(k*z) - k*Bi*exp(-k*z));
	//phi
	EzAndPhi.y = Ai*exp(k*z) +Bi*exp(-k*z);
	return EzAndPhi;
      }else{
	const double et = perm.top/perm.inside;      
	cufftComplex2 EzAndPhi;
	//TODO: bottom metallic not implemented
	//field Z
	// EzAndPhi.x = -(k*Ai*exp(k*z) - k*+Bi*exp(-k*z));
	//phi
	// EzAndPhi.y = Ai*exp(k*z) +Bi*exp(-k*z);
	return EzAndPhi;
      }
    }
    
    __device__ cufftComplex2 analyticalCorrectionTwoMetallicWalls(double k, double z, double H, Permitivity perm, MismatchVals mis){
      const auto Ai = (mis.mPH - exp(-k*H)*mis.mP0)/(exp(-k*H) - exp(k*H));
      const auto Bi = (-mis.mP0 + mis.mPH*exp(-k*H))/(1.0-exp(-2.0*k*H));
      cufftComplex2 EzAndPhi;
      //field Z
      EzAndPhi.x = -(k*Ai*exp(k*z) - k*Bi*exp(-k*z));
      //phi
      EzAndPhi.y = Ai*exp(k*z) + Bi*exp(-k*z);
      return EzAndPhi;
    }
    
    template<class Iterator>
    __device__ void fillCorrectionForWaveNumber(Permitivity perm, real k, real kmax,
						MismatchVals mis,
						cufftComplex2 linearModeCorrection,
						bool isMetallicTop, bool isMetallicBottom,
						real H, int Nz, real He, Iterator &correction){
      const real halfH = real(0.5)*H;
      for(int i = 0; i<Nz; i++){
        real z =  (halfH+real(2.0)*He)*cospi(i/real(Nz-1));
	if(abs(z) < halfH+He and k <= kmax){
	  z = z + halfH;
	  if(k>0){
	    cufftComplex2 res;
	    if(not isMetallicBottom and not isMetallicTop)
	      res = analyticalCorrectionNonMetallic(k, z, H, perm, mis);
	    else if(isMetallicBottom and isMetallicTop)
	      res = analyticalCorrectionTwoMetallicWalls(k, z, H, perm, mis);
	    else if(isMetallicBottom or isMetallicTop)
	      res = analyticalCorrectionOneMetallicWall(k, z, H, perm, mis, isMetallicTop);	    
	    correction[i] = res;
	  }
	  else{
	    correction[i].x = linearModeCorrection.x;
	    correction[i].y = linearModeCorrection.x*z +linearModeCorrection.y;
	  }
	}
	else{
	  correction[i] = cufftComplex2();
	}
      }
    }

    __global__ void computeCorrectionD(cufftComplex2 *correctionRealSpace,
				       MismatchPtr mismatch,
				       int nkx, int nky, int nz,
				       bool isMetallicBottom, bool isMetallicTop,
				       real H, real2 Lxy, real kmax,
				       real He,
				       Permitivity perm){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
      if(id >= nky*(nkx/2+1)){
	return;
      }
      IndexToWaveNumber id2wn(nkx, nky);
      WaveNumberToWaveVector wn2wv(Lxy);
      const real2 kvec = wn2wv(id2wn(id));
      const real k = sqrt(dot(kvec, kvec));
      auto corr = make_third_index_iterator(correctionRealSpace, ik.x, ik.y, Index3D(nkx/2+1, nky, 2*nz-2));
      auto mis = mismatch.fetchMisMatch(ik, {nkx, nky});
      fillCorrectionForWaveNumber(perm, k, kmax, mis, mismatch.linearModeCorrection[0],
				  isMetallicTop, isMetallicBottom, 
				  H, nz, He, corr);
    }

    namespace detail{
      __global__ void periodicExtension(cufftComplex2* signal, int nz, int nbatch){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= nbatch*nz) return;
	const int ib = id%nbatch;
	const int iz  = id/(nbatch);
	if(iz>=nz-1 or iz == 0) return;
	const int zf = 2*nz-2-iz;
	const int zi = iz;
	signal[ib + zf*nbatch]  = signal[ib + zi*nbatch];
      }

      __global__ void scaleFFTToForwardChebyshevTransform(cufftComplex2* signal, int nz, int nbatch){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>= nbatch*nz) return;
	const int iz  = id/nbatch;
	const int ib = id%nbatch;
	const int zf = 2*nz-2-iz;
	const int zi = iz;
	if(iz>0 && iz<nz-1)
	  signal[ib + zi*nbatch] += signal[ib + zf*nbatch];
	signal[ib + zi*nbatch] *= real(1.0)/real(2*nz-2);
      }

    }

    void sumCorrectionToInsideSolution(cached_vector<cufftComplex4> &correction,
				       cached_vector<cufftComplex4> &insideSolution,
				       cudaStream_t  st){
      System::log<System::DEBUG>("Sum correction to solution");
      thrust::transform(thrust::cuda::par.on(st),
			insideSolution.begin(), insideSolution.end(),
			correction.begin(), insideSolution.begin(),
			thrust::minus<cufftComplex4>());
    }

    __global__ void computeFullCorrectionXYD(const cufftComplex2* analyticalCorrection,
					     cufftComplex4* fullCorrection,
					     int nkx, int nky, int nz, real2 Lxy){
      const int id = blockDim.x*blockIdx.x + threadIdx.x;
      if(id >= (nkx/2+1)*nky)  return;
      const int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
      const real2 k = WaveNumberToWaveVector(Lxy)(IndexToWaveNumber(nkx, nky)(id));
      const auto corr = make_third_index_iterator(analyticalCorrection, ik.x, ik.y, Index3D(nkx/2+1, nky, nz));
      auto fullCorr = make_third_index_iterator(fullCorrection, ik.x, ik.y, Index3D(nkx/2+1, nky, nz));
      const bool isPairedX = ik.x != (nkx/2);
      const bool isPairedY = ik.y != (nky/2);
      fori(0, nz){
	const auto potential = corr[i].y;
	const auto Ez = corr[i].x;
	fullCorr[i].x = cufftComplex({k.x*potential.y, -k.x*potential.x})*isPairedX;
	fullCorr[i].y = cufftComplex({k.y*potential.y, -k.y*potential.x})*isPairedY;
	fullCorr[i].z = Ez;
	fullCorr[i].w = potential;
      }
    }

    cached_vector<cufftComplex4> computeFullCorrectionXY(cached_vector<cufftComplex2>& analyticalCorrection,
							 real3 boxSize, int3 n){
      System::log<System::DEBUG>("Correction XY");
      real2 Lxy = make_real2(boxSize);
      cached_vector<cufftComplex4> fullCorrection((n.x/2+1)*n.y*(2*n.z-2));
      auto analyticalCorrection_ptr = thrust::raw_pointer_cast(analyticalCorrection.data());
      auto fullCorrection_ptr = thrust::raw_pointer_cast(fullCorrection.data());
      int blockSize = 128;
      int nblocks = ((n.x/2+1)*n.y)/blockSize + 1;
      CudaCheckError();
      computeFullCorrectionXYD<<<nblocks, blockSize>>>(analyticalCorrection_ptr, fullCorrection_ptr, n.x, n.y, n.z, Lxy);
      CudaCheckError();
      return fullCorrection;
    }

    class Correction{
      Permitivity permitivity;
      int3 cellDim;
      real2 Lxy;
      real H, He;
      shared_ptr<Mismatch> mismatch;
      cufftHandle cufft_plan_forward;
      bool isMetallicBottom, isMetallicTop;
    public:
      Correction(Permitivity perm, int3 cellDim, real2 Lxy, real H, real He):
	permitivity(perm), cellDim(cellDim), Lxy(Lxy), H(H), He(He){
	this->mismatch = std::make_shared<Mismatch>(cellDim, permitivity, H, He);
	const int3 n = cellDim;
	int size = 2*n.z-2;
	int stride = 2*(n.x/2+1)*n.y;
	int dist = 1;
	int batch = 2*(n.x/2+1)*n.y;
	CufftSafeCall(cufftPlanMany(&cufft_plan_forward, 1, &size, &size,
				    stride, dist, &size, stride,
				    dist, CUFFT_Complex2Complex<real>::value, batch));
	this->isMetallicTop = std::isinf(perm.top);
	this->isMetallicBottom = std::isinf(perm.bottom);
      }
      
      ~Correction(){
	CufftSafeCall(cufftDestroy(cufft_plan_forward));
      }

      void correctSolution(cached_vector<cufftComplex4> &insideSolution,
			   cached_vector<cufftComplex4> &outsideSolution,
			   const cufftComplex2* surfaceValuesFourier, cudaStream_t st){
	System::log<System::DEBUG>("Mismatch");
	auto mismatchData = mismatch->compute(insideSolution, outsideSolution, surfaceValuesFourier, st);
	System::log<System::DEBUG>("Correction");
	real3 boxSize = make_real3(Lxy, H);
	auto analyticalCorrection = computeAnalyticalCorrectionRealSpace(mismatchData, st);
	takeAnalyticalCorrectionToChebyshevSpace(analyticalCorrection, st);
	auto fullCorrection = computeFullCorrectionXY(analyticalCorrection, boxSize, cellDim);
	sumCorrectionToInsideSolution(fullCorrection, insideSolution, st);
      }

    private:

      cached_vector<cufftComplex2> computeAnalyticalCorrectionRealSpace(MismatchPtr mis, cudaStream_t st){
	System::log<System::DEBUG>("Analytical Correction");
	const int3 n = cellDim;
	cached_vector<cufftComplex2> analyticalCorrection((n.x/2+1)*n.y*(2*n.z-2));
	auto corr = thrust::raw_pointer_cast(analyticalCorrection.data());
	//kmax must be chosen carefully so that there is no overflow in exp(kmax*HE) when evaluating the correction
	//Maximum exponents for an exponential that do not result in inf: 
#ifdef DOUBLE_PRECISION
	constexpr double maxAllowedExponent = 709.0;
#else
	constexpr float maxAllowedExponent = 88.0f;
#endif
	const real zmax = He;
	const real kmax_numerical_limit = maxAllowedExponent/zmax;
	const real h = Lxy.x/n.x;
	const real kmax = std::min(kmax_numerical_limit, real(M_PI/h));
	int blockSize = 128;
	int nblocks = ((n.x/2+1)*n.y)/blockSize + 1;
	computeCorrectionD<<<nblocks, blockSize, 0, st>>>(corr, mis, n.x, n.y, n.z,
							  isMetallicBottom, isMetallicTop,
							  H, Lxy, kmax, He, permitivity);
	CudaCheckError();
	return analyticalCorrection;
      }

      void takeAnalyticalCorrectionToChebyshevSpace(cached_vector<cufftComplex2> &analyticalCorrection,
						    cudaStream_t st){
	System::log<System::DEBUG>("Analytical Correction to Chebyshev");
	cufftComplex* d_data = (cufftComplex*) thrust::raw_pointer_cast(analyticalCorrection.data());
	const int blockSize = 128;
	const int3 n = cellDim;
	int nblocks = (((n.x/2+1))*n.y*n.z)/blockSize+1;
	detail::periodicExtension<<<nblocks, blockSize, 0, st>>>((cufftComplex2*)d_data, n.z, (n.x/2+1)*n.y);
	CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
	CufftSafeCall(cufftExecComplex2Complex<real>(cufft_plan_forward, d_data, d_data, CUFFT_FORWARD));
	nblocks = ((n.x/2 + 1)*n.y*n.z)/blockSize + 1;
	detail::scaleFFTToForwardChebyshevTransform<<<nblocks, blockSize, 0, st>>>((cufftComplex2*)d_data, n.z, (n.x/2+1)*n.y);
	CudaCheckError();
      }

    };

  }

}

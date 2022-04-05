/*Raul P. Pelaez 2020. Potential and field mismatch at the boundaries for the Doubly Periodic Poisson solver. Slab geometry
*/

#ifndef DPPOISSONSLAB_MISMATCH_CUH
#define DPPOISSONSLAB_MISMATCH_CUH
#include "global/defines.h"
#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include <vector>

namespace uammd{
  namespace DPPoissonSlab_ns{

    struct MismatchVals{
      cufftComplex mP0, mPH, mE0, mEH;
    };
    
    struct MismatchPtr{
      cufftComplex* potentialTop;
      cufftComplex* potentialBottom;
      cufftComplex* fieldZTop;
      cufftComplex* fieldZBottom;
      cufftComplex2* linearModeCorrection;
      __device__ MismatchVals fetchMisMatch(int2 ik, int2 nk){
	Index3D indexer(nk.x/2 + 1, nk.y, 1);
	MismatchVals mis;
	mis.mEH = *make_third_index_iterator(fieldZTop, ik.x, ik.y, indexer);
	mis.mE0 = *make_third_index_iterator(fieldZBottom, ik.x, ik.y, indexer);
	mis.mPH = *make_third_index_iterator(potentialTop, ik.x, ik.y, indexer);
	mis.mP0 = *make_third_index_iterator(potentialBottom, ik.x, ik.y, indexer);
	return mis;
      }
    };

    class MismatchGPU{
      int nz;
      real H;
      Permitivity perm;
      real thetaTop, thetaBottom;
      bool metallicTop = false;
      bool metallicBottom = false;
    public:
      MismatchGPU(int nz,
		  real H,
		  Permitivity perm,
		  real thetaTop, real thetaBottom):
	nz(nz),
	H(H),
	perm(perm),
	thetaTop(thetaTop), thetaBottom(thetaBottom){
	this->metallicTop = std::isinf(perm.top);
	this->metallicBottom = std::isinf(perm.bottom);
      }

      template<class Iterator1, class Iterator2>
      __device__ MismatchVals computeMismatch(Iterator1 insideSolution, Iterator2 outsideSolution,
					      cufftComplex2 surfaceValueFourier){
	auto evalthetaInside = evaluateThetas(insideSolution, thetaTop, thetaBottom, nz);
	auto evalthetaOutside = evaluateThetas(outsideSolution, thetaTop, thetaBottom, nz);
	MismatchVals mismatch;
	const real eb = real(2.0)*perm.inside/(perm.bottom + perm.inside);
	const real et = real(2.0)*perm.inside/(perm.top + perm.inside);
	if(not metallicTop){
	  const auto surfaceChargeAtTopWallFourier = surfaceValueFourier.x;
	  mismatch.mPH = evalthetaInside.y - et*evalthetaOutside.y;
	  mismatch.mEH = perm.top*et*evalthetaOutside.w - perm.inside*evalthetaInside.w - surfaceChargeAtTopWallFourier;
	}
	else{
	  const auto potentialAtTopWallFourier = surfaceValueFourier.x;
	  mismatch.mPH = evalthetaInside.y - potentialAtTopWallFourier;
	}
	if(not metallicBottom){
	  const auto surfaceChargeAtBottomWallFourier = surfaceValueFourier.y;
	  mismatch.mP0 = evalthetaInside.x - eb*evalthetaOutside.x;
	  mismatch.mE0 = perm.bottom*eb*evalthetaOutside.z - perm.inside*evalthetaInside.z + surfaceChargeAtBottomWallFourier;
	}
	else{
	  const auto potentialAtBottomWallFourier = surfaceValueFourier.y;
	  mismatch.mP0 = evalthetaInside.x - potentialAtBottomWallFourier;
	}	
	return mismatch;
      }

      template<class Iterator>
      __device__ cufftComplex2 computeLinearModeCorrection(cufftComplex mismatchFieldZBottom,
							   cufftComplex mismatchFieldZTop,
							   cufftComplex mismatchPotentialBottom,
							   cufftComplex mismatchPotentialTop,
							   Iterator outsideSolutionMode0){
	auto evalThetaOutside = evaluateThetas(outsideSolutionMode0, 0, real(M_PI), nz);
	const real et = perm.top/perm.inside;
	const real eb = perm.bottom/perm.inside;
	const auto C = real(2.0)/(et + real(1.0))*evalThetaOutside.w;
	const auto E = real(2.0)/(eb + real(1.0))*evalThetaOutside.z;
	cufftComplex A0 = cufftComplex();
	cufftComplex B0 = cufftComplex();
	if(metallicTop and not metallicBottom){	  
	  A0 = E*eb - mismatchFieldZBottom.x/perm.inside;
	  B0 = -mismatchPotentialBottom;
	}
	else if(metallicBottom and not metallicTop){
	  A0 = C*et - mismatchFieldZTop/perm.inside;
	  B0 = -mismatchPotentialBottom - A0*H;
	}
	else if(metallicTop and metallicBottom){
	  A0 = (mismatchPotentialTop - mismatchPotentialBottom)/H;
	  B0 = -mismatchPotentialBottom;
	}
	else if(not metallicTop and not metallicBottom){
	  A0 = (E*eb - mismatchFieldZBottom.x/perm.inside + C*et - mismatchFieldZTop/perm.inside)*real(0.5);
	  B0 = cufftComplex();	 
	}
	return {A0, B0};
      }

    private:
      template<class SolutionIterator>
      __device__ cufftComplex4 evaluateThetas(SolutionIterator sol, real thetaTop, real thetaBot, int Nz){
	cufftComplex2 resTop = cufftComplex2();
	cufftComplex2 resBot = cufftComplex2();
	for(int i = 0; i < Nz; i++){
	  const auto phi = sol[i].w;
	  const auto Ez = sol[i].z;
	  const real ct = cos(i*thetaTop);
	  resTop.x += phi*ct;
	  resTop.y += Ez*ct;
	  const real cb = cos(i*thetaBot);
	  resBot.x += phi*cb;
	  resBot.y += Ez*cb;
	}
	return {resBot.x, resTop.x, resBot.y, resTop.y};
      }

    };

    namespace detail{
      __device__ void storeMismatch(MismatchVals mis, MismatchPtr ptr, int2 ik, Index3D indexer){
	*make_third_index_iterator(ptr.potentialBottom, ik.x, ik.y, indexer) = mis.mP0;
	*make_third_index_iterator(ptr.potentialTop, ik.x, ik.y, indexer) = mis.mPH;
	*make_third_index_iterator(ptr.fieldZBottom, ik.x, ik.y, indexer) = mis.mE0;
	*make_third_index_iterator(ptr.fieldZTop, ik.x, ik.y, indexer) = mis.mEH;
      }
    }

    __global__ void computeMismatchD(const cufftComplex4 *insideSolution,
				     const cufftComplex4 *outsideSolution,
				     const cufftComplex2 *surfaceValuesFourier,
				     MismatchPtr mismatch,
				     MismatchGPU mcomp,
				     int nkx, int nky, int nz){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
      if(id >= nky*(nkx/2+1)){
	return;
      }
      Index3D indexer(nkx/2+1, nky, 2*nz-2);
      auto inside_ik = make_third_index_iterator(insideSolution, ik.x, ik.y, indexer);
      auto outside_ik = make_third_index_iterator(outsideSolution, ik.x, ik.y, indexer);
      cufftComplex2 sfc = *make_third_index_iterator(surfaceValuesFourier, ik.x, ik.y, indexer);
      auto m = mcomp.computeMismatch(inside_ik, outside_ik, sfc);
      detail::storeMismatch(m, mismatch, ik, indexer);
      if(id == 0){
	mismatch.linearModeCorrection[0] = mcomp.computeLinearModeCorrection(m.mE0, m.mEH, m.mP0, m.mPH, outside_ik);
      }
    }

    class Mismatch{
      using container = cached_vector<cufftComplex>;
      using container2 = cached_vector<cufftComplex2>;
      container potentialTop;
      container potentialBottom;
      container fieldZTop;
      container fieldZBottom;
      container2 linearModeCorrection;
      int3 cellDim;
      Permitivity permitivity;
      real H;
      real thetaTop, thetaBot;
    public:
      Mismatch(int3 cellDim, Permitivity perm, real H, real extraHeight):
	cellDim(cellDim), permitivity(perm), H(H){
	int numberElements = (cellDim.x/2 + 1)*cellDim.y;
	potentialTop.resize(numberElements);
	potentialBottom.resize(numberElements);
	fieldZTop.resize(numberElements);
	fieldZBottom.resize(numberElements);
	linearModeCorrection.resize(1);
	real He = extraHeight;
	real Lz = 0.5*(H + 4*He);
	this->thetaTop = acos((2*He+H)/Lz-1);
	this->thetaBot = acos((2*He)/Lz-1);
      }

      //surfaceValues hold surface charges of the walls. If the permittivity is infinite (metallic boundary) at some wall, surfaceValues hold the potential at the wall instead.
      template<class Container>
      MismatchPtr compute(Container &insideSolution, Container &outsideSolution,
			  const cufftComplex2* surfaceValuesFourier,
			  cudaStream_t st){
	System::log<System::DEBUG>("Computing mismatch");
	const auto i_ptr = thrust::raw_pointer_cast(insideSolution.data());
	const auto o_ptr = thrust::raw_pointer_cast(outsideSolution.data());
	auto mismatch_ptr = this->getRawPointers();
	int3 n = cellDim;
	MismatchGPU gpuCompute(n.z, H, permitivity, thetaTop, thetaBot);
	int blockSize = 128;
	int nblocks = ((n.x/2+1)*n.y)/blockSize + 1;
        computeMismatchD<<<nblocks, blockSize, 0, st>>>(i_ptr, o_ptr,
							surfaceValuesFourier,
							mismatch_ptr,
							gpuCompute,
							n.x, n.y, n.z);
	CudaCheckError();
	return mismatch_ptr;
      }

      MismatchPtr getRawPointers(){
	MismatchPtr ptrs;
	ptrs.potentialTop = thrust::raw_pointer_cast(potentialTop.data());
	ptrs.potentialBottom = thrust::raw_pointer_cast(potentialBottom.data());
	ptrs.fieldZTop = thrust::raw_pointer_cast(fieldZTop.data());
	ptrs.fieldZBottom = thrust::raw_pointer_cast(fieldZBottom.data());
	ptrs.linearModeCorrection = thrust::raw_pointer_cast(linearModeCorrection.data());
	return ptrs;
      }

    };

  }
}


#endif

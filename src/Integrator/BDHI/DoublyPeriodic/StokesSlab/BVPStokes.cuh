/*Raul P. Pelaez 2020-2021. Boundary Value Problem solver for DP Stokes.
 */
#ifndef BVPSTOKESSLAB_CUH
#define BVPSTOKESSLAB_CUH
#include "misc/BoundaryValueProblem/BVPSolver.cuh"
#include "utils.cuh"
#include "global/defines.h"
#include "utils/utils.h"
#include <thrust/iterator/counting_iterator.h>

namespace uammd{
  namespace DPStokesSlab_ns{
    struct BVPKernelTemporalStorage{
      BVP::StorageHandle<cufftComplex> rightHandSideChebyshevCoefficients;
      BVP::StorageHandle<cufftComplex> pressureChebyshevCoefficients;
      BVP::StorageHandle<cufftComplex> velocityChebyshevCoefficients;
      BVP::StorageHandle<cufftComplex> secondDerivativeChebyshevCoefficients;
      size_t allocationSize;
    };

    BVPKernelTemporalStorage setUpBVPKernelTemporalStorage(int numberConcurrentThreads, int nz){
      BVP::StorageRegistration mem(numberConcurrentThreads);
      BVPKernelTemporalStorage tmp;
      tmp.rightHandSideChebyshevCoefficients = mem.registerStorageRequirement<cufftComplex>(nz);
      tmp.secondDerivativeChebyshevCoefficients = mem.registerStorageRequirement<cufftComplex>(nz);
      tmp.pressureChebyshevCoefficients = mem.registerStorageRequirement<cufftComplex>(nz);
      tmp.velocityChebyshevCoefficients = mem.registerStorageRequirement<cufftComplex>(nz);
      tmp.allocationSize = mem.getRequestedStorageBytes();
      return tmp;
    }

    __device__ int2 computeWaveNumber(int id, int nkx, int nky){
      IndexToWaveNumber id2wn(nkx, nky);
      const auto waveNumber = id2wn(id);
      return waveNumber;
    }

    __device__ real2 computeWaveVector(int2 waveNumber, real2 Lxy){
      WaveNumberToWaveVector wn2wv(Lxy);
      const auto waveVector = wn2wv(waveNumber);
      return waveVector;
    }

    template<class ForceIterator>
    class RightHandSideCompute{
      const int nz;
      const real H;
      real2 waveVector;
      const ForceIterator fn;
    public:
      __device__ RightHandSideCompute(int nz, real H, real2 i_waveVector, const ForceIterator fn,
				      bool isUnpairedX, bool isUnpairedY):
	nz(nz), H(H), fn(fn), waveVector(i_waveVector){
	waveVector.x = isUnpairedX?0:waveVector.x;
	waveVector.y = isUnpairedY?0:waveVector.y;
      }

      template<class RightHandSideIterator>
      __device__ void pressure(RightHandSideIterator &rhs){
	cufftComplex df_dz_np2 = cufftComplex();
	cufftComplex df_dz_np1 = cufftComplex();
	for(int i = nz-1; i>=0; i--){
	  cufftComplex df_dz_ni = cufftComplex();
	  if(i<=nz-2){
	    df_dz_ni = df_dz_np2 + real(2.0)*(i+1)*fn[i+1].z/H;
	  }
	  if(i==0){
	    df_dz_ni *= real(0.5);
	  }
	  df_dz_np2 = df_dz_np1;
	  if(i<=nz-2){
	    df_dz_np1 = df_dz_ni;
	  }
	  const real ikf_r = -waveVector.x*fn[i].x.y - waveVector.y*fn[i].y.y;
	  const real ikf_i =  waveVector.x*fn[i].x.x + waveVector.y*fn[i].y.x;
	  rhs[i].x =  ikf_r + df_dz_ni.x;
	  rhs[i].y =  ikf_i + df_dz_ni.y;
	}
      }

      template<class RightHandSideIterator, class PressureIterator>
      __device__ void parallelVelocityX(real mu, const PressureIterator &pressure, RightHandSideIterator &rhs){
	parallelVelocity<0>(mu, pressure, rhs);
      }

      template<class RightHandSideIterator, class PressureIterator>
      __device__ void parallelVelocityY(real mu, const PressureIterator &pressure, RightHandSideIterator &rhs){
	parallelVelocity<1>(mu, pressure, rhs);
      }

      template<class RightHandSideIterator, class PressureIterator>
      __device__ void perpendicularVelocity(real mu, const PressureIterator &pressure, RightHandSideIterator &rhs){
	const real invmu = real(1.0)/mu;
	cufftComplex Dpnp2 = cufftComplex();
	cufftComplex Dpnp1 = cufftComplex();
	for(int i = nz-1; i>=0; i--){
	  cufftComplex Dpni = cufftComplex();
	  if(i<=nz-2) Dpni = Dpnp2 + real(2.0)*(i+1)*pressure[i+1]/H;
	  if(i==0) Dpni *= real(0.5);
	  rhs[i] = invmu*(Dpni - fn[i].z);
	  Dpnp2 = Dpnp1;
	  if(i<=nz-2){
	    Dpnp1 = Dpni;
	  }
	}
      }

    private:

      template<int dir, class RightHandSideIterator, class PressureIterator>
      __device__ void parallelVelocity(real mu, const PressureIterator &pressure, RightHandSideIterator &rhs){
	const real kdir = ((real*)(&waveVector))[dir];
	const real invmu = real(1.0)/mu;
	fori(0, nz){
	  const cufftComplex f = ((cufftComplex*)(&fn[i]))[dir];
	  const cufftComplex p = pressure[i];
	  rhs[i].x = invmu*(-kdir*p.y - f.x);
	  rhs[i].y = invmu*(kdir*p.x - f.y);
	}
      }

    };

    class VelocityBoundaryConditionsRightHandSide{
      cufftComplex pH;
      cufftComplex pmH;
      real kmod;
    public:

      template<class PressureIterator>
      __device__ VelocityBoundaryConditionsRightHandSide(const PressureIterator &pressure, real kmod, int nz):
	kmod(kmod){
	pH = pmH = cufftComplex();
	fori(0, nz){
	  const real sign = (i%2==0)?real(1.0):real(-1.0);
	  pH  += pressure[i];
	  pmH += sign*pressure[i];
	}
      }

      __device__ cufftComplex computeTopParallel(real kdir, real mu){
	cufftComplex alpha = cufftComplex();
	if(kmod > 0){
	  alpha =  real(-0.5)*kdir/(mu*kmod)*make_real2(-pH.y, pH.x);
	}
	return alpha;
      }

      __device__ cufftComplex computeBottomParallel(real kdir, real mu){
	cufftComplex beta = cufftComplex();
	if(kmod > 0){
	  beta = real(0.5)*kdir/(mu*kmod)*make_real2(-pmH.y, pmH.x);
	}
	return beta;
      }

      __device__ cufftComplex computeTopPerpendicular(real mu){
	cufftComplex alpha = cufftComplex();
	if(kmod > 0){
	  alpha =  real(0.5)/mu*pH;
	}
	return alpha;
      }

      __device__ cufftComplex computeBottomPerpendicular(real mu){
	cufftComplex beta = cufftComplex();
	if(kmod > 0){
	  beta =  real(0.5)/mu*pmH;
	}
	return beta;
      }

    };

    template<class BVPSolver>
    __global__ void solveBVPVelocityD(BVPSolver bvp, int nkx, int nky, int nz,
				      real2 Lxy, real H,
				      cufftComplex4* gridForce,
				      cufftComplex4* gridVelocity,
				      BVPKernelTemporalStorage tmp,
				      char* tmp_storage_raw_memory,
				      const real* precomputedVelocityChebyshevIntegrals,
				      const real* precomputedPressureChebyshevIntegrals,
				      real mu){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      const int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
      const int numberSystems = nky*(nkx/2+1);
      if(id >= numberSystems){
	return;
      }
      auto fn = make_third_index_iterator(gridForce, ik.x, ik.y, Index3D(nkx/2+1, nky, 2*nz-2));
      // if(id == 0){
      // 	fori(0, nz){
      // 	  fn[i] *= 0;
      // 	}
      // }
      const auto waveNumber = computeWaveNumber(id, nkx, nky);
      const auto waveVector = computeWaveVector(waveNumber, Lxy);
      const bool isUnpairedX = waveNumber.x == (nkx - waveNumber.x);
      const bool isUnpairedY = waveNumber.y == (nky - waveNumber.y);
      RightHandSideCompute<decltype(fn)> rhsCompute(nz, H, waveVector, fn, isUnpairedX, isUnpairedY);
      BVP::StorageRetriever stor(numberSystems, id, tmp_storage_raw_memory);
      auto rightHandSide = stor.retrieveStorage(tmp.rightHandSideChebyshevCoefficients);
      auto pressure = stor.retrieveStorage(tmp.pressureChebyshevCoefficients);
      auto an = stor.retrieveStorage(tmp.secondDerivativeChebyshevCoefficients);
      auto velocity = stor.retrieveStorage(tmp.velocityChebyshevCoefficients);
      //PRESSURE SOLVE
      rhsCompute.pressure(rightHandSide);
      {
	const auto alpha = cufftComplex();
	const auto beta = cufftComplex();
	bvp.solve(id, rightHandSide,  alpha, beta,  an, pressure);
      }
      
      //VELOCITY SOLVE
      auto gridVels = make_third_index_iterator(gridVelocity, ik.x, ik.y, Index3D(nkx/2+1, nky, 2*nz-2));
      const real k = sqrt(dot(waveVector, waveVector));
      VelocityBoundaryConditionsRightHandSide rhs_bc(pressure, k, nz);
      {
	//VELOCITY X
	rhsCompute.parallelVelocityX(mu, pressure, rightHandSide);
	{
	  const cufftComplex alpha = rhs_bc.computeTopParallel(waveVector.x, mu);
	  const cufftComplex beta = rhs_bc.computeBottomParallel(waveVector.x, mu);
	  bvp.solve(id, rightHandSide,  alpha, beta,  an, velocity);
	}
	// if(id == 0){
	//   cufftComplex linearCorrection = cufftComplex();
	//   fori(0, nz){
	//     linearCorrection += precomputedVelocityChebyshevIntegrals[i]*fn[i].x;
	//   }
	//   velocity[1] += (real(0.5)/mu)*linearCorrection;
	// }
	fori(0, nz){
	  gridVels[i].x = velocity[i];
	}
	//VELOCITY Y
	rhsCompute.parallelVelocityY(mu, pressure, rightHandSide);
	{
	  const cufftComplex alpha = rhs_bc.computeTopParallel(waveVector.y, mu);
	  const cufftComplex beta = rhs_bc.computeBottomParallel(waveVector.y, mu);
	  bvp.solve(id, rightHandSide,  alpha, beta,  an, velocity);
	}
	// if(id == 0){
	//   cufftComplex linearCorrection = cufftComplex();
	//   fori(0, nz){
	//     linearCorrection += precomputedVelocityChebyshevIntegrals[i]*fn[i].y;
	//   }
	//   velocity[1] += (real(0.5)/mu)*linearCorrection;
	// }
	fori(0, nz){
	  gridVels[i].y = velocity[i];
	}
      }
      //VELOCITY Z
      rhsCompute.perpendicularVelocity(mu, pressure, rightHandSide);
      {
	const cufftComplex alpha = rhs_bc.computeTopPerpendicular(mu);
	const cufftComplex beta = rhs_bc.computeBottomPerpendicular(mu);
	bvp.solve(id, rightHandSide, alpha, beta,  an, velocity);
      }
      if(id == 0){
	cufftComplex linearCorrection = cufftComplex();
	fori(0, nz){
	  linearCorrection += precomputedPressureChebyshevIntegrals[i]*fn[i].z;
	}
	pressure[0] += real(0.5)*linearCorrection;
	pressure[1] += real(0.5)*linearCorrection;
      }

      fori(0, nz){
	gridVels[i].z = velocity[i];
	gridVels[i].w = pressure[i];
      }
    }
  }
}
#endif

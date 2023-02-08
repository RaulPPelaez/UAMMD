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
      BVP::StorageHandle<complex> rightHandSideChebyshevCoefficients;
      BVP::StorageHandle<complex> pressureChebyshevCoefficients;
      BVP::StorageHandle<complex> velocityChebyshevCoefficients;
      BVP::StorageHandle<complex> secondDerivativeChebyshevCoefficients;
      size_t allocationSize;
    };

    BVPKernelTemporalStorage setUpBVPKernelTemporalStorage(int numberConcurrentThreads, int nz){
      BVP::StorageRegistration mem(numberConcurrentThreads);
      BVPKernelTemporalStorage tmp;
      tmp.rightHandSideChebyshevCoefficients = mem.registerStorageRequirement<complex>(nz);
      tmp.secondDerivativeChebyshevCoefficients = mem.registerStorageRequirement<complex>(nz);
      tmp.pressureChebyshevCoefficients = mem.registerStorageRequirement<complex>(nz);
      tmp.velocityChebyshevCoefficients = mem.registerStorageRequirement<complex>(nz);
      tmp.allocationSize = mem.getRequestedStorageBytes();
      return tmp;
    }

    //Computes the right hand side for the different PDEs (velocity xyz and pressure)
    class RightHandSideCompute{
      const int nz;
      const real H;
      real2 waveVector;
      const real mu;
    public:
      __device__ RightHandSideCompute(int nz, real H, real2 i_waveVector, real mu,
				      bool isUnpairedX, bool isUnpairedY):
	nz(nz), H(H), mu(mu), waveVector(i_waveVector){
	//ik differentiation is 0 in unpaired modes
	waveVector.x = isUnpairedX?0:waveVector.x;
	waveVector.y = isUnpairedY?0:waveVector.y;
      }

      template<class ForceIterator, class RightHandSideIterator>
      __device__ void pressure(ForceIterator &fxn, ForceIterator &fyn, ForceIterator &fzn,
			       RightHandSideIterator &rhs){
	complex df_dz_np2 = complex();
	complex df_dz_np1 = complex();
	for(int i = nz-1; i>=0; i--){
	  complex df_dz_ni = complex();
	  if(i<=nz-2){
	    df_dz_ni = df_dz_np2 + real(2.0)*(i+1)*fzn[i+1]/H;
	  }
	  if(i==0){
	    df_dz_ni *= real(0.5);
	  }
	  df_dz_np2 = df_dz_np1;
	  if(i<=nz-2){
	    df_dz_np1 = df_dz_ni;
	  }
	  const real ikf_r = -waveVector.x*fxn[i].y - waveVector.y*fyn[i].y;
	  const real ikf_i =  waveVector.x*fxn[i].x + waveVector.y*fyn[i].x;
	  rhs[i].x =  ikf_r + df_dz_ni.x;
	  rhs[i].y =  ikf_i + df_dz_ni.y;
	}
      }

      template<class ForceIterator, class RightHandSideIterator, class PressureIterator>
      __device__ void parallelVelocityX(const ForceIterator &fxn,
					const PressureIterator &pressure,
					RightHandSideIterator &rhs){
	parallelVelocity<0>(fxn, pressure, rhs);
      }

      template<class ForceIterator, class RightHandSideIterator, class PressureIterator>
      __device__ void parallelVelocityY(const ForceIterator &fyn,
					const PressureIterator &pressure, RightHandSideIterator &rhs){
	parallelVelocity<1>(fyn, pressure, rhs);
      }

      template<class ForceIterator, class RightHandSideIterator, class PressureIterator>
      __device__ void perpendicularVelocity(const ForceIterator &fzn,
					    const PressureIterator &pressure, RightHandSideIterator &rhs){
	const real invmu = real(1.0)/mu;
	complex Dpnp2 = complex();
	complex Dpnp1 = complex();
	for(int i = nz-1; i>=0; i--){
	  complex Dpni = complex();
	  if(i<=nz-2) Dpni = Dpnp2 + real(2.0)*(i+1)*pressure[i+1]/H;
	  if(i==0) Dpni *= real(0.5);
	  rhs[i] = invmu*(Dpni - fzn[i]);
	  Dpnp2 = Dpnp1;
	  if(i<=nz-2){
	    Dpnp1 = Dpni;
	  }
	}
      }

    private:

    template<int dir, class ForceIterator, class RightHandSideIterator, class PressureIterator>
      __device__ void parallelVelocity(const ForceIterator &fn,
				       const PressureIterator &pressure, RightHandSideIterator &rhs){
	const real kdir = ((real*)(&waveVector))[dir];
	const real invmu = real(1.0)/mu;
	fori(0, nz){
	  const complex f = fn[i];
	  const complex p = pressure[i];
	  rhs[i].x = invmu*(-kdir*p.y - f.x);
	  rhs[i].y = invmu*(kdir*p.x - f.y);
	}
      }

    };

    //Computes the right hand side for the velocity BCs.
    class VelocityBoundaryConditionsRightHandSide{
      complex pH;
      complex pmH;
      real kmod;
    public:

      template<class PressureIterator>
      __device__ VelocityBoundaryConditionsRightHandSide(const PressureIterator &pressure, real kmod, int nz):
	kmod(kmod){
	pH = pmH = complex();
	fori(0, nz){
	  const real sign = (i%2==0)?real(1.0):real(-1.0);
	  pH  += pressure[i];
	  pmH += sign*pressure[i];
	}
      }

      __device__ complex computeTopParallel(real kdir, real mu){
	complex alpha = complex();
	if(kmod > 0){
	  alpha =  real(-0.5)*kdir/(mu*kmod)*make_real2(-pH.y, pH.x);
	}
	return alpha;
      }

      __device__ complex computeBottomParallel(real kdir, real mu){
	complex beta = complex();
	if(kmod > 0){
	  beta = real(0.5)*kdir/(mu*kmod)*make_real2(-pmH.y, pmH.x);
	}
	return beta;
      }

      __device__ complex computeTopPerpendicular(real mu){
	complex alpha = complex();
	if(kmod > 0){
	  alpha =  real(0.5)/mu*pH;
	}
	return alpha;
      }

      __device__ complex computeBottomPerpendicular(real mu){
	complex beta = complex();
	if(kmod > 0){
	  beta =  real(0.5)/mu*pmH;
	}
	return beta;
      }

    };

    template<class BVPSolver>
    __global__ void solveBVPVelocityD(BVPSolver bvp, int nkx, int nky, int nz,
				      real2 Lxy, real H, //Domain is [-H, H]
				      const DataXYZPtr<complex> gridForce,
				      FluidPointers<complex> fluid,
				      BVPKernelTemporalStorage tmp,
				      char* tmp_storage_raw_memory,
				      const real* precomputedVelocityChebyshevIntegrals,
				      const real* precomputedPressureChebyshevIntegrals,
				      real mu,
				      WallMode mode){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      const int2 ik = make_int2(id%(nkx/2+1), id/(nkx/2+1));
      const int numberSystems = nky*(nkx/2+1);
      if(id >= numberSystems){
	return;
      }
      auto indexer = Index3D(nkx/2+1, nky, 2*nz-2);
      auto fxn = make_third_index_iterator(gridForce.x(), ik.x, ik.y, indexer);
      auto fyn = make_third_index_iterator(gridForce.y(), ik.x, ik.y, indexer);
      auto fzn = make_third_index_iterator(gridForce.z(), ik.x, ik.y, indexer);
      auto velx = make_third_index_iterator(fluid.velocityX, ik.x, ik.y, indexer);
      auto vely = make_third_index_iterator(fluid.velocityY, ik.x, ik.y, indexer);
      auto velz = make_third_index_iterator(fluid.velocityZ, ik.x, ik.y, indexer);
      auto pressure = make_third_index_iterator(fluid.pressure, ik.x, ik.y, indexer);
      //Zero mode is added as a correction when there are walls
      if(id == 0){
	//If there are no walls the zero mode is just zero
	//In other modes we need the forces stored in the zero mode, so do not overwrite.
	if(true or mode == WallMode::none){
	  fori(0, nz){
	    velx[i] *= 0;
	    vely[i] *= 0;
	    velz[i] *= 0;
	    pressure[i] *= 0;
	  }
	}
	return;
      }
      const auto waveNumber = computeWaveNumber(id, nkx, nky);
      const auto waveVector = computeWaveVector(waveNumber, Lxy);
      //ik differentiation is 0 in unpaired modes
      const bool isUnpairedX = waveNumber.x == (nkx - waveNumber.x);
      const bool isUnpairedY = waveNumber.y == (nky - waveNumber.y);
      RightHandSideCompute rhsCompute(nz, H, waveVector, mu, isUnpairedX, isUnpairedY);
      BVP::StorageRetriever stor(numberSystems, id, tmp_storage_raw_memory);
      auto rightHandSide = stor.retrieveStorage(tmp.rightHandSideChebyshevCoefficients);
      //      auto pressure = stor.retrieveStorage(tmp.pressureChebyshevCoefficients);
      auto an = stor.retrieveStorage(tmp.secondDerivativeChebyshevCoefficients);
      //      auto velocity = stor.retrieveStorage(tmp.velocityChebyshevCoefficients);
      //PRESSURE SOLVE
      rhsCompute.pressure(fxn, fyn, fzn, rightHandSide);
      {
	const auto alpha = complex();
	const auto beta = complex();
	bvp.solve(id, rightHandSide,  alpha, beta,  an, pressure);
      }

      //VELOCITY SOLVE
      const real k = sqrt(dot(waveVector, waveVector));
      VelocityBoundaryConditionsRightHandSide rhs_bc(pressure, k, nz);
      {
	//VELOCITY X
	rhsCompute.parallelVelocityX(fxn, pressure, rightHandSide);
	{
	  const complex alpha = rhs_bc.computeTopParallel((not isUnpairedX)*waveVector.x, mu);
	  const complex beta = rhs_bc.computeBottomParallel((not isUnpairedX)*waveVector.x, mu);
	  bvp.solve(id, rightHandSide,  alpha, beta,  an, velx);
	}
	//VELOCITY Y
	rhsCompute.parallelVelocityY(fyn, pressure, rightHandSide);
	{
	  const complex alpha = rhs_bc.computeTopParallel((not isUnpairedY)*waveVector.y, mu);
	  const complex beta = rhs_bc.computeBottomParallel((not isUnpairedY)*waveVector.y, mu);
	  bvp.solve(id, rightHandSide,  alpha, beta,  an, vely);
	}
      }
      //VELOCITY Z
      rhsCompute.perpendicularVelocity(fzn, pressure, rightHandSide);
      {
	const complex alpha = rhs_bc.computeTopPerpendicular(mu);
	const complex beta = rhs_bc.computeBottomPerpendicular(mu);
	bvp.solve(id, rightHandSide, alpha, beta,  an, velz);
      }
      //PRESSURE LINEAR CORRECTION, only in open boundaries case
      if(mode == WallMode::none and id == 0){
	complex linearCorrection = complex();
	fori(0, nz){
	  linearCorrection += precomputedPressureChebyshevIntegrals[i]*fzn[i];
	}
	pressure[0] += real(0.5)*linearCorrection;
	pressure[1] += real(0.5)*linearCorrection;
      }
    }
  }
}
#endif

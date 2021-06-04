/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver.
 */

#ifndef DOUBLYPERIODIC_STOKES_CUH
#define DOUBLYPERIODIC_STOKES_CUH

#include"Integrator/BDHI/DoublyPeriodic/StokesSlab/utils.cuh"
#include"Integrator/Integrator.cuh"


#include "utils/utils.h"
#include "global/defines.h"
#include"misc/BoundaryValueProblem/BVPSolver.cuh"
#include"misc/ChevyshevUtils.cuh"
#include"StokesSlab/FastChebyshevTransform.cuh"
#include"StokesSlab/utils.cuh"
#include"StokesSlab/Correction.cuh"

namespace uammd{
  namespace DPStokesSlab_ns{
    struct Gaussian{
      int3 support;
      Gaussian(real tolerance, real width, real h, real H, int supportxy, int nz):H(H), nz(nz){
	this-> prefactor = cbrt(pow(2*M_PI*width*width, -1.5));
	this-> tau = -1.0/(2.0*width*width);
	rmax = supportxy*h*0.5;
	support = {supportxy, supportxy, supportxy};
	int ct = int(nz*(acos(-2*(-H*0.5+rmax)/H)/M_PI));
	support.z = 2*ct+1;
      }
      
      inline __host__  __device__ int3 getMaxSupport() const{
	return support;
      }

      inline __host__  __device__ int3 getSupport(int3 cell) const{
	real ch = real(-0.5)*H*cospi((real(cell.z))/(nz-1));
	real zmax = thrust::min(ch+rmax, H*real(0.5));
	int czt = int((nz)*(acos(real(-2.0)*(zmax)/H)/real(M_PI)));
	real zmin = thrust::max(ch-rmax, -H*real(0.5));
	int czb = int((nz)*(acos(real(-2.0)*(zmin)/H)/real(M_PI)));
	int sz = 2*thrust::max(czt-cell.z, cell.z-czb)+1;
	return make_int3(support.x, support.y, thrust::min(sz, support.z));
      }
      
      inline __host__  __device__ real phiX(real r) const{
	return prefactor*exp(tau*r*r);
      }
      
      inline __host__  __device__ real phiY(real r) const{
	return prefactor*exp(tau*r*r);
      }
      //For this algorithm we spread a particle and its image to enforce the force density outside the slab is zero.
      //A particle on the wall position will spread zero force. phi(r) = phi(r) - phi(r_img);
      inline __host__  __device__ real phiZ(real r, real3 pi) const{
	if(fabs(r) >=rmax){
	  return 0;
	}
	else{
	  real top_rimg =  H-2*pi.z+r;
	  real bot_rimg = -H-2*pi.z+r;
	  real rimg = thrust::min(fabs(top_rimg), fabs(bot_rimg));
	  real phi_img = rimg>=rmax?real(0.0):prefactor*exp(tau*rimg*rimg);
	  real phi = prefactor*exp(tau*r*r);
	  return phi - phi_img;
	}	 
      }

      // inline __host__  __device__ real phi(real r, real3 rij, real3 pi) const{
      // 	 if(fabs(r) >=rmax){
      // 	   return 0;
      // 	 }
      // 	 else{
      // 	   real phi = prefactor*prefactor*prefactor*exp(tau*rij.x*rij.x)*exp(tau*rij.y*rij.y);
      // 	   real top_rimg =  H-2*pi.z+rij.z;
      // 	   real bot_rimg = -H-2*pi.z+rij.z;
      // 	   real rimg = thrust::min(fabs(top_rimg), fabs(bot_rimg));
      // 	   real phiz_img = rimg>=rmax?real(0.0):exp(tau*rimg*rimg);
      // 	   return phi*(exp(tau*rij.z*rij.z) - phiz_img);
      // 	 }
	 
      // }

    private:
      real prefactor;
      real tau;
      real H;
      real rmax;
      int nz;
    };

    //Tabulated parameters for the BM kernel, this map returns, for a given support, beta and the normalization
    //BM_params[w] -> {beta, norm}
    static const std::map<int, double2> BM_params({ {4, {1.785,1.471487792829305}},
						    {5, {1.886,1.456086423397354}}});
    struct BarnettMagland{
      int3 support;      
      real beta;
      real alpha;
      real norm;

      BarnettMagland(real tolerance_ignored, real width_ignored, real h, real H, int supportxy, int nz):
	H(H), nz(nz){
	this->rmax = supportxy*h*0.5*2;
	support = {supportxy+1, supportxy+1, supportxy+10};
	int ct = int(nz*(acos(-2*(-H*0.5+rmax)/H)/M_PI));
	support.z = 2*ct+1;
	this->alpha = supportxy*0.5*h;
	//If the support is not tabulated an exception will be thrown
	try{
	  this->beta = supportxy*BM_params.at(supportxy).x;
	  this->norm = BM_params.at(supportxy).y;
	}
	catch(...){
	  System::log<System::EXCEPTION>("BM kernel: Untabulated support: %d", supportxy);
	  throw std::runtime_error("BM kernel: Untabulated support");
	}
      }
      
      inline __host__  __device__ int3 getMaxSupport() const{
	return support;
      }

      inline __host__  __device__ int3 getSupport(int3 cell) const{
	real ch = real(-0.5)*H*cospi((real(cell.z))/(nz-1));
	real zmax = thrust::min(ch+rmax, H*real(0.5));
	int czt = int((nz)*(acos(real(-2.0)*(zmax)/H)/real(M_PI)));
	real zmin = thrust::max(ch-rmax, -H*real(0.5));
	int czb = int((nz)*(acos(real(-2.0)*(zmin)/H)/real(M_PI)));
	int sz = 2*thrust::max(czt-cell.z, cell.z-czb)+1;
	return make_int3(support.x, support.y, thrust::min(sz, support.z));
      }
      
      inline __host__  __device__ real bm(real r) const{
        real za  = fabs(r/alpha);
	return za>=1?0:exp(beta*(sqrt(1-za*za)-1))/norm;
      }
      
      inline __host__  __device__ real phiX(real r) const{
	return bm(r);
      }
      
      inline __host__  __device__ real phiY(real r) const{
	return bm(r);
      }
      //For this algorithm we spread a particle and its image to enforce the force density outside the slab is zero.
      //A particle on the wall position will spread zero force. phi(r) = phi(r) - phi(r_img);
      inline __host__  __device__ real phiZ(real r, real3 pi) const{
	real top_rimg =  H-2*pi.z+r;
	real bot_rimg = -H-2*pi.z+r;
	real rimg = thrust::min(fabs(top_rimg), fabs(bot_rimg));
	real phi_img = bm(rimg);
	real phi = bm(r);
	return phi - phi_img;
      }
    private:
      real H;
      real rmax;
      int nz;
    };

    class DPStokes{
    public:
      //using Kernel = Gaussian;
      using Kernel = BarnettMagland;      
      using Grid = chebyshev::doublyperiodic::Grid;
      using QuadratureWeights = chebyshev::doublyperiodic::QuadratureWeights;

      struct Parameters{
	int nxy, nz;
	real dt;
	real viscosity;
	real Lxy;
	real H;
	real tolerance = 1e-7;
	real gw;
	int support = -1; //-1 means auto compute from tolerance
	//Can be either none, bottom or slit
	WallMode mode = WallMode::none;
      };

      DPStokes(Parameters par);
    
      ~DPStokes(){
	System::log<System::MESSAGE>("[DPStokes] Destroyed");
      }

      cached_vector<real4> Mdot(real4* pos, real4* forces, int N, cudaStream_t st);
      
    private:
      shared_ptr<Kernel> kernel;
      shared_ptr<FastChebyshevTransform> fct;
      shared_ptr<Correction> correction;
      
      gpu_container<real> zeroModeVelocityChebyshevIntegrals;
      gpu_container<real> zeroModePressureChebyshevIntegrals;
    
      void setUpGrid(Parameters par);
      void initializeKernel(int supportxy);
      void printStartingMessages(Parameters par);
      void resizeVectors();
      void initializeBoundaryValueProblemSolver();
      void initializeQuadratureWeights();
      void precomputeIntegrals();
      void resetGridForce();
      void tryToResetGridForce();
      cached_vector<real4> spreadForces(real4* pos, real4* forces, int N, cudaStream_t st);
      void solveBVPVelocity(cached_vector<cufftComplex4> &gridDataFourier, cudaStream_t st);
      void resizeTmpStorage(size_t size);
      cached_vector<real4> interpolateVelocity(cached_vector<real4> &gridData, real4* pos,
					       int N, cudaStream_t st);
    
      real Lxy;
      real H;
      Grid grid;   
      real viscosity;
      real gw;
      real tolerance;
      WallMode mode;
      shared_ptr<QuadratureWeights> qw;
      shared_ptr<BVP::BatchedBVPHandler> bvpSolver;

    };
  }
}
#include"StokesSlab/initialization.cu"
#include"StokesSlab/DPStokes.cu"
#endif


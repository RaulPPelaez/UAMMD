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
namespace uammd{
  namespace DPStokesSlab_ns{
    struct Gaussian{
      int support;
      Gaussian(real tolerance, real width, real h, real H, int supportxy, int nz):H(H), nz(nz){
	this-> prefactor = cbrt(pow(2*M_PI*width*width, -1.5));
	this-> tau = -1.0/(2.0*width*width);
	rmax = supportxy*h*0.5;
	support = supportxy;
      }
      
      inline __host__  __device__ int3 getSupport(int3 cell) const{
	real ch = real(0.5)*H*cospi((real(cell.z))/(nz-1));
	int czt = int((nz)*(acos(real(2.0)*(ch+rmax)/H)/real(M_PI)));
	int czb = int((nz)*(acos(real(2.0)*(ch-rmax)/H)/real(M_PI)));
	int sz = 2*thrust::max(cell.z - czt, czb - cell.z)+1;
	return make_int3(support, support, sz);
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
	real top_rimg = H-pi.z-r;
	real bot_rimg = -H-pi.z-r;
	real rimg = thrust::min(abs(top_rimg), abs(bot_rimg));
	return (abs(r)>=rmax)?real(0):(prefactor*(exp(tau*r*r)-(rimg>=rmax?real(0.0):exp(tau*rimg*rimg))));
      }

    private:
      real prefactor;
      real tau;
      real H;
      real rmax;
      int nz;
    };
          
    class DPStokes{
    public:
      using Kernel = Gaussian;      
      using Grid = chebyshev::doublyperiodic::Grid;
      using QuadratureWeights = chebyshev::doublyperiodic::QuadratureWeights;

      struct Parameters{
	int3 cells = make_int3(-1, -1, -1); //Number of Fourier nodes in each direction
	real dt;
	real viscosity;
	Box box;
	real tolerance = 1e-7;
	real gw;
	int support = -1; //-1 means auto compute from tolerance
      };

      DPStokes(Parameters par);
    
      ~DPStokes(){
	System::log<System::MESSAGE>("[DPStokes] Destroyed");
      }

      cached_vector<real4> Mdot(real4* pos, real4* forces, int N, cudaStream_t st);
      
    private:
      shared_ptr<Kernel> kernel;
      shared_ptr<FastChebyshevTransform> fct;
      
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
      cached_vector<real4> interpolateVelocity(cached_vector<real4> &gridData, real4* pos, int N, cudaStream_t st);
    
      Box box;
      Grid grid;   
      real viscosity;
      real gw;
      real tolerance;
    
      shared_ptr<QuadratureWeights> qw;
      shared_ptr<BVP::BatchedBVPHandler> bvpSolver;

    };
  }
}

#include"StokesSlab/initialization.cu"
#include"StokesSlab/DPStokes.cu"
#endif


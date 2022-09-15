
/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver.
 */

#ifndef DOUBLYPERIODIC_STOKES_CUH
#define DOUBLYPERIODIC_STOKES_CUH

#include"Integrator/BDHI/DoublyPeriodic/StokesSlab/utils.cuh"
#include"Integrator/Integrator.cuh"


#include "misc/IBM_kernels.cuh"
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
      Gaussian(real tolerance, real width, real h, real H, int supportxy, int nz, bool torqueMode = false):H(H), nz(nz){
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

    private:
      real prefactor;
      real tau;
      real H;
      real rmax;
      int nz;
    };

    //Tabulated parameters for the BM kernel and its derivative, this map returns, for a given support, beta and the normalization
    //BM_params[w] -> {beta, norm}
    //TODO: the norm should be autocomputed
    //TODO: Use BM from IBM_kernels directly
    // static const std::map<int, double2> BM_params({ {4, {1.785,1.471487792829305}},
    // 						    {5, {1.886,1.456086423397354}},
    // 						    {6, {1.3267,1.46442666831683}}});
    // //{6, {1.714,1.452197522749675}}});

    // static const std::map<int, double2> BM_torque_params({{6, {2.216,1.157273283795062}}});

    // //The BM kernel and its derivative
    // inline __host__  __device__ real bm(real r, real alpha, real beta, real norm){
    //   real za  = fabs(r/alpha);
    //   return za>=real(1.0)?real(0.0):exp(beta*(sqrt(real(1.0)-za*za)-real(1.0)))/norm;
    // }

    // inline __host__  __device__ real bm_deriv(real r, real alpha, real beta, real norm){
    //   real a2 = alpha*alpha;
    //   return -beta*r*bm(r,alpha, beta, norm)/(a2*sqrt(real(1.0)-r*r/a2));
    // }

    // namespace detail{
    //   real suggestBetaBM(real w){
    // 	throw std::runtime_error("Beta<0 Not implemented yet, please give me a value");
    // 	return 0;
    //   }
    // }
    //[1] Taken from https://arxiv.org/pdf/1712.04732.pdf
    struct BarnettMagland{
      IBM_kernels::BarnettMagland bm;
      real a;
      int3 support;

      BarnettMagland(real w, real beta, real i_alpha, real h, real H, int nz):
	H(H), nz(nz), bm(i_alpha, beta){
	int supportxy = w+0.5;
	this->rmax = w*h*0.5;
	support.x = support.y = supportxy;
	int ct = ceil((nz-1)*(acos((H*0.5-rmax)/(0.5*H))/M_PI));
	support.z = 2*ct+1;
	this->a = h;
	System::log<System::MESSAGE>("BM kernel: beta: %g, alpha: %g, w: %g", beta, i_alpha, w);
      }

      inline __host__  __device__ int3 getMaxSupport() const{
	return support;
      }

      inline __host__  __device__ int3 getSupport(real3 pos, int3 cell) const{
	real bound = H*real(0.5);
	real ztop = thrust::min(pos.z+rmax, bound);
	real zbot = thrust::max(pos.z-rmax, -bound);
	int czb = int((nz-1)*(acos(ztop/bound)/real(M_PI)));
	int czt = int((nz-1)*(acos(zbot/bound)/real(M_PI)));
	int sz = 2*thrust::max(cell.z - czb, czt - cell.z)+1;
	return make_int3(support.x, support.y, sz);
      }

      inline __host__  __device__ real phiX(real r, real3 pi) const{
	return bm.phi(r/a)/a;
      }

      inline __host__  __device__ real phiY(real r, real3 pi) const{
	return bm.phi(r/a)/a;
      }
      //For this algorithm we spread a particle and its image to enforce the force density outside the slab is zero.
      //A particle on the wall position will spread zero force. phi(r) = phi(r) - phi(r_img);
      inline __host__  __device__ real phiZ(real r, real3 pi) const{
	real top_rimg =  H-real(2.0)*pi.z+r;
	real bot_rimg = -H-real(2.0)*pi.z+r;
	real rimg = thrust::min(fabs(top_rimg), fabs(bot_rimg));
	real phi_img = bm.phi(rimg/a)/a;
	real phi = bm.phi(r/a)/a;
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
      //A different kernel can be used for spreading forces and torques
      using KernelTorque = BarnettMagland;
      using Grid = chebyshev::doublyperiodic::Grid;
      using QuadratureWeights = chebyshev::doublyperiodic::QuadratureWeights;
      using WallMode = WallMode;
      //Parameters, -1 means that it will be autocomputed if not present
      struct Parameters{
	int nx, ny;
	int nz = -1;
	real dt;
	real viscosity;
	real Lx, Ly;
	real H;
	real tolerance = 1e-7;
	real w, w_d;
	real hydrodynamicRadius;
	real beta = -1;
	real beta_d = -1;
	real alpha = -1;
	real alpha_d = -1;
	//Can be either none, bottom or slit
	WallMode mode = WallMode::none;
      };

      DPStokes(Parameters par);

      ~DPStokes(){
	System::log<System::MESSAGE>("[DPStokes] Destroyed");
      }

      //Computes the velocities and angular velocities given the forces and torques
      std::pair<cached_vector<real3>, cached_vector<real3>>
      Mdot(real4* pos, real4* forces, real4* torque, int N, cudaStream_t st = 0);

      //Computes the velocities given the forces
      cached_vector<real3> Mdot(real4* pos, real4* forces, int N, cudaStream_t st = 0);

    private:
      shared_ptr<Kernel> kernel;
      shared_ptr<KernelTorque> kernelTorque;
      shared_ptr<FastChebyshevTransform> fct;
      shared_ptr<Correction> correction;

      gpu_container<real> zeroModeVelocityChebyshevIntegrals;
      gpu_container<real> zeroModePressureChebyshevIntegrals;

      void setUpGrid(Parameters par);
      void initializeKernel(Parameters par);
      void printStartingMessages(Parameters par);
      void resizeVectors();
      void initializeBoundaryValueProblemSolver();
      void initializeQuadratureWeights();
      void precomputeIntegrals();
      void resetGridForce();
      void tryToResetGridForce();
      cached_vector<real4> spreadForces(real4* pos, real4* forces, int N, cudaStream_t st);
      void addSpreadTorquesFourier(real4* pos, real4* torques, int N,
				   cached_vector<cufftComplex4> &gridData, cudaStream_t st);
      void solveBVPVelocity(cached_vector<cufftComplex4> &gridDataFourier, cudaStream_t st);
      void resizeTmpStorage(size_t size);
      cached_vector<real3> interpolateVelocity(cached_vector<real4> &gridData, real4* pos,
					       int N, cudaStream_t st);
      cached_vector<cufftComplex4> computeGridAngularVelocityCheb(cached_vector<cufftComplex4> &gridVelsCheb,
						      cudaStream_t st);
      cached_vector<real3> interpolateAngularVelocity(cached_vector<real4> &gridData,
						      real4* pos, int N, cudaStream_t st);
      real Lx, Ly;
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

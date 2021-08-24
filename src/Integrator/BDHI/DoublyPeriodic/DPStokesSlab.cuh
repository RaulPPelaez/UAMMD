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
    static const std::map<int, double2> BM_params({ {4, {1.785,1.471487792829305}},
						    {5, {1.886,1.456086423397354}},
						    {6, {1.3267,1.46442666831683}}});
    //{6, {1.714,1.452197522749675}}});
    
    static const std::map<int, double2> BM_torque_params({{6, {2.216,1.157273283795062}}});

    //The BM kernel and its derivative
    inline __host__  __device__ real bm(real r, real alpha, real beta, real norm){
      real za  = fabs(r/alpha);
      return za>=1?0:exp(beta*(sqrt(1-za*za)-1))/norm;
    }

    inline __host__  __device__ real bm_deriv(real r, real alpha, real beta, real norm){
      real a2 = alpha*alpha;
      return -beta*r*bm(r,alpha, beta, norm)/(a2*sqrt(real(1.0)-r*r/a2));
    }

    namespace detail{
      template<class Foo>
      real integrate(Foo foo, real rmin, real rmax, int Nr){
	double integral = foo(rmin)*0.5;
	for(int i = 1; i<Nr; i++){
	  integral += foo(rmin+i*(rmax-rmin)/Nr);
	}
	integral += foo(rmax)*0.5;
	integral *= (rmax-rmin)/(real)Nr;
	return integral;
      }
      real suggestBetaBM(real w){
	throw std::runtime_error("Beta<0 Not implemented yet, please give me a value");
	return 0;
      }
    }
    //[1] Taken from https://arxiv.org/pdf/1712.04732.pdf
    struct BarnettMagland{
      int3 support;      
      real beta;
      real alpha;
      real norm;
      real computeNorm() const{
	auto foo=[=](real r){return bm(r, alpha, beta, 1.0);};
	real norm = detail::integrate(foo, -alpha, alpha, 100000);
	return norm;
      }

      BarnettMagland(real w, real beta, real i_alpha, real h, real H, int nz):
      //real tolerance_ignored, real width_ignored, real h, real H, int supportxy, int nz, bool torqueMode = false):
	H(H), nz(nz), beta(beta), alpha(i_alpha){
	int supportxy = w/h+0.5;
	this->rmax = w;
	support.x = support.y = supportxy+1;
	int ct = int(nz*(acos(-2*(-H*0.5+rmax)/H)/M_PI));
	support.z = 2*ct+1;
	if(alpha>w*h*0.5){
	  throw std::runtime_error("BM: alpha has to be less or equal to w*h/2 ("+std::to_string(w*h*0.5)+"), found" +std::to_string(alpha));
	}
	if(alpha<0) this->alpha = w*h*0.5;
	if(beta<0) this->beta = detail::suggestBetaBM(w);
	this->norm = this->computeNorm();
	System::log<System::MESSAGE>("BM kernel: beta: %g, alpha: %g, w: %g, norm: %g", beta, alpha, w, norm);
	//this->alpha = supportxy*0.5*h;	
	//If the support is not tabulated an exception will be thrown
	// try{
	//   if(torqueMode){
	//     this->beta = supportxy*BM_torque_params.at(supportxy).x;
	//     this->norm = BM_torque_params.at(supportxy).y;
	//   }
	//   else{
	//     this->beta = supportxy*BM_params.at(supportxy).x;
	//     this->norm = BM_params.at(supportxy).y;
	//   }
	// }
	// catch(...){
	//   System::log<System::EXCEPTION>("BM kernel: Untabulated support: %d", supportxy);
	//   throw std::runtime_error("BM kernel: Untabulated support");
	// }
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
	return bm(r,alpha, beta, norm);
      }
      
      inline __host__  __device__ real phiY(real r) const{
	return bm(r,alpha, beta, norm);
      }
      //For this algorithm we spread a particle and its image to enforce the force density outside the slab is zero.
      //A particle on the wall position will spread zero force. phi(r) = phi(r) - phi(r_img);
      inline __host__  __device__ real phiZ(real r, real3 pi) const{
	real top_rimg =  H-real(2.0)*pi.z+r;
	real bot_rimg = -H-real(2.0)*pi.z+r;
	real rimg = thrust::min(fabs(top_rimg), fabs(bot_rimg));
	real phi_img = bm(rimg,alpha, beta, norm);
	real phi = bm(r,alpha, beta, norm);
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
      Mdot(real4* pos, real4* forces, real4* torque, int N, cudaStream_t st);

      //Computes the velocities given the forces
      cached_vector<real3> Mdot(real4* pos, real4* forces, int N, cudaStream_t st);

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


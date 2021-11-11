/*Raul P. Pelaez 2018-2020. Force Coupling Method BDHI Module

  The implementation of the algorithm is in FCM/FCM_impl.cuh

  This code implements the algorithm described in [1], using cuFFT to solve te
velocity in eq. 24 of [1] and compute the brownian fluctuations of eq. 30 in [1]
(it only needs two FFT's). It only includes the stokeslet terms.

  The operator terminology used in the comments (as well as the wave space part
of the algorithm) comes from [2], the PSE basic reference.

You can choose different Kernels by changing the "using Kernel" below. A bunch of them are available in FCM_kernels.cuh
  References:
  [1] Fluctuating force-coupling method for simulations of colloidal suspensions. Eric E. Keaveny. 2014.
  [2]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations. Fiore, Balboa, Donev and Swan. 2017.
*/
#ifndef BDHI_FCM_CUH
#define BDHI_FCM_CUH
#include "uammd.cuh"
#include "Integrator/Integrator.cuh"
#include "BDHI.cuh"
#include"FCM/FCM_impl.cuh"

namespace uammd{
  namespace BDHI{
    class FCM{
      using Kernel = FCM_ns::Kernels::Gaussian;
      using KernelTorque = FCM_ns::Kernels::GaussianTorque;
      using FCM_super = FCM_impl<Kernel, KernelTorque>;      
      std::shared_ptr<FCM_super> fcm;
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
      shared_ptr<System> sys;

    public:
      using Parameters = FCM_super::Parameters;
      
      FCM(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,
	  Parameters par):
	pd(pd), pg(pg), sys(sys){
	if(par.seed == 0)
	  par.seed = sys->rng().next32();
	this->fcm = std::make_shared<FCM_super>(par);
      }

      void setup_step(cudaStream_t st = 0){}

      /*Compute M·F and B·dW in Fourier space
      σ = dx*dy*dz; h^3 in [1]
      Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
      = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
    */
      void computeMF(real3* MF, cudaStream_t st = 0){
	sys->log<System::DEBUG1>("[BDHI::FCM] Computing MF....");
	auto force = pd->getForce(access::location::gpu, access::mode::read);
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	int numberParticles = pg->getNumberParticles();
        auto disp = fcm->computeHydrodynamicDisplacements(pos.begin(), force.begin(), nullptr,
							  numberParticles, st);
	thrust::copy(thrust::cuda::par.on(st),
		     disp.first.begin(), disp.first.end(),
		     MF);
      }

      void computeBdW(real3* BdW, cudaStream_t st = 0){
	//This part is included in Fourier space when computing MF
      }

      void finish_step(cudaStream_t st = 0){}

    };

    class FCMIntegrator: public Integrator{
      using Kernel = FCM_ns::Kernels::Gaussian;
      using KernelTorque = FCM_ns::Kernels::GaussianTorque;
      using FCM_super = FCM_impl<Kernel, KernelTorque>;
      std::shared_ptr<FCM_super> fcm;

    public:
      using Parameters = FCM_super::Parameters;

      FCMIntegrator(shared_ptr<ParticleData> pd,
		    shared_ptr<ParticleGroup> pg,
		    Parameters par):
	FCMIntegrator(pd, pg, pd->getSystem(), par){}

      FCMIntegrator(shared_ptr<ParticleData> pd, Parameters par):
	FCMIntegrator(pd, std::make_shared<ParticleGroup>(pd, "All"), par){}

      FCMIntegrator(shared_ptr<ParticleData> pd,
		    shared_ptr<System> sys,
		    Parameters par):
	FCMIntegrator(pd, par){}

      FCMIntegrator(shared_ptr<ParticleData> pd,
		    shared_ptr<ParticleGroup> pg,
		    shared_ptr<System> sys,
		    Parameters par):
	Integrator(pd, pg,"BDHI::FCMIntegrator"),
	dt(par.dt){
	if(par.seed == 0)
	  par.seed = sys->rng().next32();
	this->fcm = std::make_shared<FCM_super>(par);
	cudaStreamCreate(&st);
      }

      ~FCMIntegrator(){
	cudaStreamDestroy(st);
      }
      
      void forwardTime() override;

      auto getFCM_impl(){
	return fcm;
      }
      
    private:
      uint steps = 0;
      cudaStream_t st;
      real dt;
      void updateInteractors();
      void resetForces();
      void resetTorques();
      auto computeHydrodynamicDisplacements();
      void computeCurrentForces();      
    };
  }
}

#include "BDHI_FCM.cu"
#endif

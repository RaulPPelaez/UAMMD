/*Raul P. Pelaez 2018-2020. Force Coupling Method BDHI Module

  This code implements the algorithm described in [1], using cuFFT to solve te
velocity in eq. 24 of [1] and compute the brownian fluctuations of eq. 30 in [1]
(it only needs two FFT's). It only includes the stokeslet terms.

  This code is adapted from PSE, basically the factor sinc(ka/2)^2 is removed
from the kernel and the near part is removed.

  The operator terminology used in the comments (as well as the wave space part
of the algorithm) comes from [2], the PSE basic reference.

You can choose different Kernels by changing the "using Kernel" below. A bunch of them are available in FCM_kernels.cuh
  References:
  [1] Fluctuating force-coupling method for simulations of colloidal suspensions. Eric E. Keaveny. 2014.
  [2]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations. Fiore, Balboa, Donev and Swan. 2017.

Contributors:
Pablo Palacios - 2021: Introduce the torques functionality.
*/
#ifndef BDHI_FCM_CUH
#define BDHI_FCM_CUH
#include "uammd.cuh"
#include "Integrator/Integrator.cuh"
#include "BDHI.cuh"
#include"utils/quaternion.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftComplex3.cuh"
#include "utils/container.h"
#include "utils/Grid.cuh"
#include "FCM_kernels.cuh"

namespace uammd{
  namespace BDHI{
#ifdef UAMMD_DEBUG
    template<class T> using gpu_container = thrust::device_vector<T>;
    template<class T>  using cached_vector = detail::UninitializedCachedContainer<T>;
#else
    template<class T> using gpu_container = thrust::device_vector<T, managed_allocator<T>>;
    template<class T> using cached_vector = thrust::device_vector<T, managed_allocator<T>>;
#endif
    
    class FCM: public Integrator{
    public:
      using Kernel = FCM_ns::Kernels::Gaussian;
      //using Kernel = FCM_ns::Kernels::BarnettMagland;
      //using Kernel = FCM_ns::Kernels::Peskin::threePoint;
      //using Kernel = FCM_ns::Kernels::Peskin::fourPoint;
      //using Kernel = FCM_ns::Kernels::GaussianFlexible::sixPoint;
      using KernelTorque =  FCM_ns::Kernels::GaussianTorque;
      using cufftComplex = cufftComplex_t<real>;
      using cufftComplex3 = cufftComplex3_t<real>;

      struct Parameters: BDHI::Parameters{
	int3 cells = make_int3(-1, -1, -1); //Number of Fourier nodes in each direction
	int steps = 0;
	uint seed;
      };

      FCM(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,
	  Parameters par);

      FCM(shared_ptr<ParticleData> pd,
	  shared_ptr<System> sys,
	  Parameters par):
	FCM(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, par){}

      ~FCM();

      void setup_step(              cudaStream_t st = 0){}
      void computeMF(real3* MF,     cudaStream_t st = 0);
      void computeBdW(real3* BdW,   cudaStream_t st = 0);
      void finish_step(             cudaStream_t st = 0){}
      void forwardTime() override;
      
      real getHydrodynamicRadius(){
	return hydrodynamicRadius;
      }

      real getSelfMobility(){
	//O(a^8) accuracy. See Hashimoto 1959.
	//With a Gaussian this expression has a minimum deviation from measuraments of 7e-7*rh at L=64*rh.
	//The translational invariance of the hydrodynamic radius however decreases arbitrarily with the tolerance.
	//Seems that this deviation decreases with L, so probably is due to the correction below missing something.
	long double rh = this->getHydrodynamicRadius();
	long double L = box.boxSize.x;
	long double a = rh/L;
	long double a2= a*a; long double a3 = a2*a;
	long double c = 2.83729747948061947666591710460773907l;
	long double b = 0.19457l;
	long double a6pref = 16.0l*M_PIl*M_PIl/45.0l + 630.0L*b*b;
	return  1.0l/(6.0l*M_PIl*viscosity*rh)*(1.0l-c*a+(4.0l/3.0l)*M_PIl*a3-a6pref*a3*a3);
      }

      //Computes the velocities given the forces
      template<typename vtype> void Mdot(real3 *Mv, vtype *v, cudaStream_t st);

      //Computes the velocities and angular velocities given the forces and torques
      std::pair<cached_vector<real3>, cached_vector<real3>>
      Mdot(real4* pos, real4* force, real4* torque, cudaStream_t st);

    private:
      
      cudaStream_t st;
      uint seed;

      real temperature;
      real dt;
      real viscosity;
      real hydrodynamicRadius;
      int steps;
      
      std::shared_ptr<Kernel> kernel;
      std::shared_ptr<KernelTorque> kernelTorque;
      
      Box box;
      Grid grid;

      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<char> cufftWorkArea;
      //thrust::device_vector<real3> K; /*Shear 3x3 matrix*/

      Parameters par;
      
      void initializeGrid(Parameters par);
      void initializeKernel(Parameters par);
      void initializeKernelTorque(Parameters par);
      void printMessages(Parameters par);
      void initCuFFT();
      
      cached_vector<real3> spreadForces(real4* pos, real4* force, cudaStream_t st);
      cached_vector<cufftComplex3> forwardTransform(cached_vector<real3>& gridReal, cudaStream_t st);
      void addSpreadTorquesFourier(real4* pos, real4* torque, cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st);
      void convolveFourier(cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st);
      void addBrownianNoise(cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st);
      cached_vector<real3> inverseTransform(cached_vector<cufftComplex3>& gridFourier, cudaStream_t st);
      cached_vector<real3> interpolateVelocity(real4* pos, cached_vector<real3>& gridVels, cudaStream_t st);
      void interpolateVelocity(real4* pos, real3* linearVelocities, cached_vector<real3>& gridVels, cudaStream_t st);
      cached_vector<cufftComplex3> computeGridAngularVelocityFourier(cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st);
      cached_vector<real3> interpolateAngularVelocity(real4* pos, cached_vector<real3>& gridAngVels, cudaStream_t st);

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

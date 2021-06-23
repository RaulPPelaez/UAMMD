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
*/
#ifndef BDHI_FCM_CUH
#define BDHI_FCM_CUH
#include"uammd.cuh"
#include "BDHI.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftComplex3.cuh"
#include"utils/Grid.cuh"
#include "FCM_kernels.cuh"

namespace uammd{
  namespace BDHI{
    class FCM{
    public:
      //Choose a different kernel by uncommenting the line
      using Kernel = FCM_ns::Kernels::Gaussian;
      //using Kernel = FCM_ns::Kernels::BarnettMagland;
      //using Kernel = FCM_ns::Kernels::Peskin::threePoint;
      //using Kernel = FCM_ns::Kernels::Peskin::fourPoint;
      //using Kernel = FCM_ns::Kernels::GaussianFlexible::sixPoint;

      using cufftComplex = cufftComplex_t<real>;
      using cufftComplex3 = cufftComplex3_t<real>;

      struct Parameters: BDHI::Parameters{
	int3 cells = make_int3(-1, -1, -1); //Number of Fourier nodes in each direction
      };

      FCM(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,
	  Parameters par);

      ~FCM();

      void setup_step(              cudaStream_t st = 0){}
      void computeMF(real3* MF,     cudaStream_t st = 0);
      void computeBdW(real3* BdW,   cudaStream_t st = 0);
      void finish_step(             cudaStream_t st = 0){}

      real getHydrodynamicRadius(){
	return hydrodynamicRadius;
      }

      real getCellSize(){
	return grid.cellSize.x;
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

    private:
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
      shared_ptr<System> sys;
      uint seed;

      real temperature;
      real dt;
      real viscosity;
      real hydrodynamicRadius;

      std::shared_ptr<Kernel> kernel;

      Box box;

      Grid grid;

      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<char> cufftWorkArea;

      thrust::device_vector<cufftComplex> gridVelsFourier;
      thrust::device_vector<real3> gridVels;

      template<typename vtype>
      void Mdot(real3 *Mv, vtype *v, cudaStream_t st);

      void initializeGrid(Parameters par);
      void initializeKernel(Parameters par);
      void printMessages(Parameters par);

      void initCuFFT();
      template<typename vtype>
      void spreadForces(vtype *v, cudaStream_t st);
      void forwardTransformForces(cudaStream_t st);
      void convolveFourier(cudaStream_t st);
      void addBrownianNoise(cudaStream_t st);
      void inverseTransformVelocity(cudaStream_t st);
      void interpolateVelocity(real3 *Mv, cudaStream_t st);

    };
  }
}

#include "BDHI_FCM.cu"
#endif

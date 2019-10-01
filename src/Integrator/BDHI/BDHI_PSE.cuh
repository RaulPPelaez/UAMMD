/*Raul P. Pelaez 2017. Positively Split Edwald BDHI Module

  As this is a BDHI module. BDHI_PSE computes the terms M·F and B·dW in the differential equation:
            dR = K·R·dt + M·F·dt + sqrt(2Tdt)· B·dW

  The mobility, M, is computed according to the Rotne-Prager-Yamakawa (RPY) tensor.

  The computation uses periodic boundary conditions (PBC)
  and partitions the RPY tensor in two, positively defined contributions [1], so that:
      M = Mr + Mw
       Mr - A real space short range contribution.
       Mw - A wave space long range contribution.

  Such as:
     M·F = Mr·F + Mw·F
     B·dW = sqrt(Mr)·dWr + sqrt(Mw)·dWw
####################      Short Range     #########################


  Mr·F: The short range contribution of M·F is computed using a neighbour list (this is like a sparse matrix-vector product in which each element is computed on the fly), see PSE_ns::RPYNearTransverser.
        The RPY near part function (see Apendix 1 in [1]) is precomputed and stored in texture memory,
	see PSE_ns::RPYPSE_nearTextures.

  sqrt(Mr)·dW: The near part stochastic contribution is computed using the Lanczos algorithm (see misc/LanczosAlgorithm.cuh), the function that computes M·v is provided via a functor called PSE_ns::Dotctor, the logic of M·v itself is the same as in M·F (see PSE_ns::RPYNearTransverser) and is computed with the same neighbour list.

###################        Far range     ###########################



  Mw·F:  Mw·F = σ·St·FFTi·B·FFTf·S · F. The long range wave space part.
         -σ: The volume of a grid cell
	 -S: An operator that spreads each element of a vector to a regular grid using a gaussian kernel.
	 -FFT: Fast fourier transform operator.
	 -B: A fourier scaling factor in wave space to transform forces to velocities, see eq.9 in [1].

        Related functions:
	FFT: cufftExecR2C (forward), cufftC2R(inverse)
	S: PSE_ns::particles2Grid (S), PSE_ns::grid2Particles (σ·St)
	B: PSE_ns::fillFourierScalingFactor, PSE_ns::forceFourier2vel

  sqrt(Mw)·dWw: The far range stochastic contribution is computed in fourier space along M·F as:
               Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
                            = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	        Only one St·FFTi is needed, the stochastic term is added as a velocity in fourier space.                dWw is a gaussian random vector of complex numbers, special care must be taken to ensure the correct conjugacy properties needed for the FFT. See PSE_ns::fourierBrownianNoise

Therefore, in the case of Mdot_far, for computing M·F, Bw·dWw is also summed.

computeBdW computes only the real space stochastic contribution.

References:

[1]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations
           -  https://arxiv.org/pdf/1611.09322.pdf
[2]  Spectral accuracy in fast Ewald-based methods for particle simulations
           -  http://www.sciencedirect.com/science/article/pii/S0021999111005092

See BDHI_PSE.cu for more info.
 */
#ifndef BDHI_PSE_CUH
#define BDHI_PSE_CUH

#include "BDHI.cuh"
#include "utils/utils.h"
#include"misc/TabulatedFunction.cuh"
#include "global/defines.h"
#include"Interactor/NeighbourList/CellList.cuh"
#include"misc/LanczosAlgorithm.cuh"
#include<cufft.h>
#include<thread>
#include"utils/Grid.cuh"

#ifndef SINGLE_PRECISION
#define cufftComplex cufftDoubleComplex
#define cufftReal cufftDoubleReal
#define cufftExecR2C cufftExecD2Z
#define cufftExecC2R cufftExecZ2D
#define CUFFT_C2R CUFFT_Z2D
#define CUFFT_R2C CUFFT_D2Z

#endif

namespace uammd{
  namespace BDHI{

    class PSE{
    public:

      struct Parameters: BDHI::Parameters{
	//Splitting parameter, works best between 0.5 and 1.0
	//lower values will give more importance to the near part (neighbour list) and higher values will
	// put the weight of the computation in the far part (FFT).
	real psi = 0.5;
      };
      PSE(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,
	  Parameters par);
      ~PSE();
      void setup_step(              cudaStream_t st = 0);
      void computeMF(real3* MF,     cudaStream_t st = 0);
      void computeBdW(real3* BdW,   cudaStream_t st = 0);
      void computeDivM(real3* divM, cudaStream_t st = 0);
      void finish_step(             cudaStream_t st = 0);


      template<typename vtype>
      void Mdot_near(real3 *Mv, vtype *v, cudaStream_t st);
      template<typename vtype>
      void Mdot_far(real3 *Mv, vtype *v, cudaStream_t st);
      template<typename vtype>
      void Mdot(real3 *Mv, vtype *v, cudaStream_t st);


      real getHydrodynamicRadius(){
	return hydrodynamicRadius;
      }
      real getSelfMobility(){
	return M0;
      }

    private:
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
      shared_ptr<System> sys;
      //std::thread Mdot_nearThread, Mdot_farThread, NearNoise_Thread;

      ullint seed;

      real hydrodynamicRadius;

      real temperature;
      real dt;
      real M0;
      real psi; /*Splitting factor*/

      /****Near (real space) part *****/
      /*Rodne Prager Yamakawa PSE near real space part textures*/
      thrust::device_vector<real2> tableDataRPY; //Storage for tabulatedFunction
      shared_ptr<TabulatedFunction<real2>> RPY_near;

      Box box;
      real rcut;
      real lanczosTolerance;
      curandGenerator_t curng;
      shared_ptr<CellList> cl;
      shared_ptr<LanczosAlgorithm> lanczos;

      /****Far (wave space) part) ******/
      real kcut; /*Wave space cutoff and corresponding real space grid size */
      Grid grid; /*Wave space Grid parameters*/

      /*Grid interpolation kernel parameters*/
      int3 P; //Gaussian spreading/interpolation kernel support points in each direction (total support=2*P+1)*/
      real3 m, eta; // kernel width and gaussian splitting in each direction

      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<char> cufftWorkArea; //Work space for cufft

      thrust::device_vector<cufftComplex> gridVelsFourier; //Interpolated grid forces/velocities in fourier space
      thrust::device_vector<real3> fourierFactor;  // Fourier scaing factors to go from F to V in wave space

      cudaStream_t stream, stream2;
    };
  }
}

#include "BDHI_PSE.cu"
#endif

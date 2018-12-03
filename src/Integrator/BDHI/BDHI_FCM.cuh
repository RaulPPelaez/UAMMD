/*Raul P. Pelaez 2018. Force Coupling Method BDHI Module

  This code implements the algorithm described in [1], using cuFFT to solve te velocity in eq. 24 of [1] and compute the brownian fluctuations of eq. 30 in [1] (it only needs two FFT's). It only includes the stokeslet terms.

  This code is adapted from PSE, basically the factor sinc(ka/2)^2 is removed from the kernel and the near part is removed. Also the spreading/interpolation kernel is now an exponential with different support and std.

  The operator terminology used in the comments (as well as the wave space part of the algorithm) comes from [2], the PSE basic reference.
  References:
  [1] Fluctuating force-coupling method for simulations of colloidal suspensions. Eric E. Keaveny. 2014.
  [2]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations. Fiore, Balboa, Donev and Swan. 2017.
*/
#ifndef BDHI_FCM_CUH
#define BDHI_FCM_CUH

#include "BDHI.cuh"
#include "utils/utils.h"
#include "global/defines.h"
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
    namespace FCM_ns{      
      /*A convenient struct to pack 3 complex numbers, that is 6 real numbers*/
      struct cufftComplex3{
	cufftComplex x,y,z;
      };

      inline __device__ __host__ cufftComplex3 operator+(const cufftComplex3 &a, const cufftComplex3 &b){
	return {a.x + b.x, a.y + b.y, a.z + b.z};
      }
      inline __device__ __host__ void operator+=(cufftComplex3 &a, const cufftComplex3 &b){
	a.x += b.x; a.y += b.y; a.z += b.z;
      }
    }
    
    class FCM{
    public:
      using cufftComplex3 = FCM_ns::cufftComplex3;
      struct Parameters: BDHI::Parameters{
	int3 cells = make_int3(-1, -1, -1); //Number of Fourier nodes in each direction
      };
      FCM(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,
	  Parameters par);
      ~FCM();
      void setup_step(              cudaStream_t st = 0);
      void computeMF(real3* MF,     cudaStream_t st = 0);    
      void computeBdW(real3* BdW,   cudaStream_t st = 0);  
      void computeDivM(real3* divM, cudaStream_t st = 0);
      void finish_step(             cudaStream_t st = 0);

      template<typename vtype>
      void Mdot_far(real3 *Mv, vtype *v, cudaStream_t st);
      template<typename vtype>
      void Mdot(real3 *Mv, vtype *v, cudaStream_t st);

    private:
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
      shared_ptr<System> sys;
      ullint seed;
      
      real temperature;
      real dt;
      real viscosity;
    
      Box box;

      /****Far (wave space) part) ******/
      Grid grid; /*Wave space Grid parameters*/
      
      /*Grid interpolation kernel parameters*/
      int3 P; //Gaussian spreading/interpolation kernel support points in each direction (total support=2*P+1)*/
      real sigma; //Gaussian kernel std
      
      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<char> cufftWorkArea; //Work space for cufft
      
      thrust::device_vector<cufftComplex> gridVelsFourier; //Interpolated grid forces/velocities in fourier/real space

      cudaStream_t stream, stream2;

      void initCuFFT();
      template<typename vtype>
      void spreadParticles(vtype *v, cudaStream_t st);
      void convolveFourier(cudaStream_t st);
      void interpolateParticles(real3 *Mv, cudaStream_t st);

    };
  }
}

#include "BDHI_FCM.cu"
#endif

/*Raul P. Pelaez 2018. Force Coupling Method BDHI Module
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
      //std::thread Mdot_nearThread, Mdot_farThread, NearNoise_Thread;

      ullint seed;
      
      real temperature;
      real dt;
      real viscosity;
    
      /****Near (real space) part *****/    
      /*Rodne Prager Yamakawa FCM near real space part textures*/
      thrust::device_vector<real2> tableDataRPY; //Storage for tabulatedFunction

      Box box;

      /****Far (wave space) part) ******/
      Grid grid; /*Wave space Grid parameters*/
      
      /*Grid interpolation kernel parameters*/
      int3 P; //Gaussian spreading/interpolation kernel support points in each direction (total support=2*P+1)*/
      real sigma;
      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<real> cufftWorkArea; //Work space for cufft
      
      thrust::device_vector<cufftComplex> gridVelsFourier; //Interpolated grid forces/velocities in fourier space
      thrust::device_vector<real3> fourierFactor;  // Fourier scaing factors to go from F to V in wave space

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

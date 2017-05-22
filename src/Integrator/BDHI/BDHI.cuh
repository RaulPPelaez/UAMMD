/*Raul P. Pelaez 2017. Brownian Dynamics with Hydrodynamic Interactions virtual method.
  
All BDHI methods compute the same terms to solve the differential equation:

  dR = (K·R + M·F)·dt + sqrt(2·dt·T)·B·dW + T·div(M)·dt

This base class allows to code different BDHI methods just by providing functions to compute MF, Bdw and div(M).

 */

#ifndef BDHI_CUH
#define BDHI_CUH
#include<cuda.h>
#include<curand.h>
#include"utils/utils.h"
#include"globals/globals.h"
#include"BDHI_common.cuh"
#include"utils/cuda_lib_defines.h"
#include"utils/utils.h"
namespace BDHI{
  class BDHI_Method{
  public:
    BDHI_Method(): M0(0), rh(0), N(0){}
    void init(real M0, real rh, int N){
      this->M0 = M0;
      this->rh = rh;
      this->N = N;
      this->init();
    }
    void init(){
      /*Create noise*/
      curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(curng, grng.next());
      /*Create a temporal vector to warm up curand*/
      GPUVector3 noise(N);
      //Curand fill with gaussian numbers with mean 0 and var 1
      /*This shit is obscure, curand will only work with an even number of elements*/
      curandGenerateNormal(curng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
    }
    
    BDHI_Method(real M0, real rh, int N):
      M0(M0), rh(rh), N(N){
      this->init();
    }
    ~BDHI_Method(){
      curandDestroyGenerator(curng);
    }

    virtual void setup_step(              cudaStream_t st = 0){}
    virtual void computeMF(real3* MF,     cudaStream_t st = 0) = 0;
    virtual void computeBdW(real3* BdW,   cudaStream_t st = 0) = 0;
    virtual void computeDivM(real3* divM, cudaStream_t st = 0) = 0;
    virtual void finish_step(              cudaStream_t st = 0){}
    curandGenerator_t getRNG(){return curng;}
  protected:
    curandGenerator_t curng;
    real M0;
    real rh;
    uint BLOCKSIZE = 128; /*CUDA kernel block size, threads per block*/    
    int N;
  };
  
}
#endif

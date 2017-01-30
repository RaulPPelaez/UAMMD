//WARNING: DIFFUSION REFERS TO MOBILITY M = D/kT
/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics integrator GPU kernels and callers

  Functions to update the positions according to:
  
  R += (K·R + M·F)·dt + (2·T·dt)^1/2 · B·dW


*/
#include "globals/defines.h"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include"BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<thrust/device_ptr.h>
#include<thrust/device_vector.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>

#include<iostream>


using namespace thrust;
using namespace std;
#define TPB 1024
namespace brownian_hy_euler_maruyama_ns{
  __constant__ Params params;



  void initGPU(Params m_params){
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }

  /*Fills a real3 array by ignoring the .w element in a real4 array*/
  __global__ void real4_to_real3D(const real4 __restrict__ *v4, real3 __restrict__ *v3, uint N){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    v3[i] = make_real3(v4[i]);
  
  }

  void real4_to_real3GPU(real4 *v4, real3* v3, uint N){
    uint nthreads = TPB<N?TPB:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0); 
    
    real4_to_real3D<<<nthreads,nblocks>>>(v4, v3, N);
  }

  /*T=0 case is templated*/
  template<bool noise>
  __global__ void integrateGPUD(real4 __restrict__  *pos,
				const real3 __restrict__  *MF,
				const real3 __restrict__ *BdW,
				const real3 __restrict__ *K,
				const real3 __restrict__ *divM){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=params.N) return;
    bool D2 = params.L.z == real(0.0);
    /*Position and color*/
    real4 pc = pos[i];
    real3 p = make_real3(pc);
    real c = pc.w;

    /*Shear stress*/
    real3 KR = make_real3(0);
    KR.x = dot(K[0], p);
    KR.y = dot(K[1], p);
    /*2D clause. Although K[2] should be 0 in 2D anyway...*/
    if(!D2)
      KR.z = dot(K[2], p);
    

    /*Update the position*/
    p += (KR + MF[i])*params.dt;
    /*T=0 is treated specially, there is no need to produce noise*/
    if(noise){
      real3 bdw  = BdW[i];
      if(params.L.z == real(0.0))
	bdw.z = 0;
      p += params.sqrt2Tdt*bdw;
    }

    if(D2 && divM){
      real3 divm = divM[i];
      divm.z = real(0.0);
      //p += params.T*divm*params.invDelta*params.invDelta*params.dt; //For RFD
      p += params.T*params.dt*divm;
    }       
    
    /*Write to global memory*/
    pos[i] = make_real4(p,c);
  }

  //Update the positions
  void integrateGPU(real4 *pos,
		    real3* MF, real3* BdW, real3* divM,
		    real3* K, uint N){
    uint nthreads = TPB<N?TPB:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0);
    /*Noise switch is templated*/
    if(!BdW)
      integrateGPUD<false><<<nblocks, nthreads>>>(pos, MF, BdW, K, divM);
    else{
      integrateGPUD<true><<<nblocks, nthreads>>>(pos, MF, BdW, K, divM);
    }
  }   
}
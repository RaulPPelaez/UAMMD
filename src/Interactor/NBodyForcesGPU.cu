/*
  Raul P. Pelaez 2016. NBody Force Interactor GPU kernels

  Computes the interaction between all pairs in the system. Currently only gravitational force
  The shared memory layout in the computeForces kernel is taken from "The CUDA handbook" by Nicholas Wilt

TODO:
100-computeForces doesnt work right, some particles act up.
100-Tweak Nblocks and Nthreads in computeForce 
90- Allow custom force

*/

#include"NBodyForcesGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"

namespace nbody_ns{
  __device__ real4 forceij(real4 posi, real4 posj){
    real3 r12 = make_real3(posj-posi);
    real r2 = dot(r12, r12)+real(0.01);
    real invr = rsqrtf(r2);
    real invr3 = invr*invr*invr;
    real4 force = make_real4(posj.w*invr3*r12, real(0.0));
    return make_real4(posj.w*invr3*r12, real(0.0));
  }


  /*There is some problem here*/
  __global__ void  computeNBodyForceD(real4 *force, real4 *pos, uint N){
    extern __shared__ real4 shPos[];
    uint index = blockIdx.x*blockDim.x + threadIdx.x;
    for(uint i=index; i<N;  i+=blockDim.x*gridDim.x){
      real4 fi = make_real4(real(0.0));    
      real4 posi = pos[i];
      for(uint j = 0; j<N; j += blockDim.x){
	shPos[threadIdx.x] = pos[j+threadIdx.x];
	__syncthreads();
	for(size_t k = 0; k<blockDim.x; k++){
	  real4 posj = shPos[k];
	  fi += forceij(posi, posj);
	}
	__syncthreads();
      }
      force[i] = fi;
    
    }

  }



  void computeNBodyForce(real4 *force, real4 *pos, uint N){
    computeNBodyForceD<<<128*32, 128, 128*sizeof(real4)>>>(force, pos, N);
  }

}

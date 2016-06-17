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

__device__ float4 forceij(float4 posi, float4 posj){
  float3 r12 = make_float3(posj-posi);
  float r2 = dot(r12, r12)+0.01f;
  float invr = rsqrtf(r2);
  float invr3 = invr*invr*invr;
  float4 force = make_float4(posj.w*invr3*r12, 0.0f);
  return make_float4(posj.w*invr3*r12, 0.0f);
}


/*There is some problem here*/
__global__ void  computeNBodyForceD(float4 *force, float4 *pos, uint N){
  extern __shared__ float4 shPos[];
  uint index = blockIdx.x*blockDim.x + threadIdx.x;
  for(uint i=index; i<N;  i+=blockDim.x*gridDim.x){
     float4 fi = make_float4(0.0f);    
     float4 posi = pos[i];
     for(uint j = 0; j<N; j += blockDim.x){
       shPos[threadIdx.x] = pos[j+threadIdx.x];
       __syncthreads();
       for(size_t k = 0; k<blockDim.x; k++){
     	float4 posj = shPos[k];
     	fi += forceij(posi, posj);
       }
       __syncthreads();
     }
    force[i] = fi;
    
  }

}



void computeNBodyForce(float4 *force, float4 *pos, uint N){
  computeNBodyForceD<<<128*32, 128, 128*sizeof(float4)>>>(force, pos, N);
}

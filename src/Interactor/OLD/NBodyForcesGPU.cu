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

#define TPB 128
namespace nbody_ns{
  inline __device__ real3 forceij(const real4 &posi, const real4 &posj){
    real3 r12 = make_real3(posj)-make_real3(posi);
    real r2 = dot(r12, r12);
    if(r2==real(0.0))
      return make_real3(0.0);
    real invr = rsqrtf(r2);
    if(r2>real(1.0)){
      r2 *= real(4.0);
      return make_real3(real(0.75)*real(2.0)*(r2-real(2.0))/(r2*r2)*r12*invr);
    }
    else{
      //return make_real3(0.0);
      return make_real3(real(0.09375)*real(2.0)*r12*invr);
    }

 //    real3 r12 = make_real3(posj)-make_real3(posi);
 //    real r2 = dot(r12, r12)+real(0.01);
 //     real r6 = r2*r2*r2;
 // #ifdef SINGLE_PRECISION
 //     real invr3 = rsqrtf(r6);
 // #else
 //     real invr3 = rsqrt(r6);
 // #endif
 //     return make_real4(invr3*r12, real(0.0));
    
  }



  __global__ void computeForceGPUD(const __restrict__ real4  *pos,
				   __restrict__ real4 *force,
				   int numTiles, /*Thread paralellism level, 
						   controls how many elements are stored in 
						   shared memory and
						   computed in parallel between synchronizations*/
				   uint N){
    /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
    uint id = blockIdx.x*blockDim.x+threadIdx.x;
    /*Each thread handles the interaction between particle id and all the others*/
    /*Storing blockDim.x positions in shared memory and processing all of them in parallel*/
    extern __shared__ real4 shPos[];
    
    const real4 pi = pos[id]; /*My position*/
    real3 f = make_real3(real(0.0)); /*The three elements of the result D·v I compute*/

    /*Distribute the N particles in numTiles tiles.
      Storing in each tile blockDim.x positions and elements of v in shared memory*/
    /*This way all threads are accesing the same memory addresses at the same time*/
    for(int tile = 0; tile<numTiles; tile++){
      /*Save this tile pos to shared memory*/
      shPos[threadIdx.x] = pos[tile*blockDim.x+threadIdx.x];
      __syncthreads();
      /*Go through all the particles in the current tile*/
      #pragma unroll 8
      for(uint counter = 0; counter<blockDim.x; counter++){
	int cur_j = tile*blockDim.x+counter;
	if(id != cur_j && cur_j<N)
	  f += forceij(pi,shPos[counter]);	
      }/*End of particles in tile loop*/
      __syncthreads();

    }/*End of tile loop*/
    /*Write the result to global memory*/
    force[id] += make_real4(f,0);
  }

  void computeNBodyForce(real4 *force, real4 *pos, uint N){

    int Nblocks  = (N+TPB-1)/TPB;
    int numtiles = (N + TPB-1)/TPB;

    computeForceGPUD<<<Nblocks, TPB, TPB*sizeof(real4)>>>(pos, force, numtiles, N);
  }

}

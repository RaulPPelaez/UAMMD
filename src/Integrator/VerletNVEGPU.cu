/*Raul P. Pelaez 2016. Two step velocity VerletNVE Integrator GPU callers 

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
TODO:
100- K sum could be done better with some thrust magic
*/
#include"globals/defines.h"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include "VerletNVEGPU.cuh"
#include<thrust/device_ptr.h>
#include<thrust/reduce.h>
#include<thrust/transform_reduce.h>
#include<iostream>
using namespace std;
using namespace thrust;
#define TPB 128

namespace verlet_nve_ns{
  __constant__ Params params;
  
  void initGPU(Params m_params){
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }


  /*Integrate the movement*/
  //All the parameters in Params struct are available here
  __global__ void integrateGPUD(real4 __restrict__  *pos,
				real3 __restrict__ *vel,
				const real4 __restrict__  *force,
				int step){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=params.N) return;

    /*Half step velocity*/
    vel[i] += make_real3(force[i])*params.dt*0.5f;
    
    if(params.L.z==0.0f) vel[i].z = 0.0f; //2D
    
    /*In the first step, upload positions*/
    if(step==1)
      pos[i] += make_real4(vel[i])*params.dt;
    
    
  }

  //Update the positions
  void integrateGPU(real4 *pos, real3 *vel, real4 *force, uint N, int step){
    
    uint nthreads = TPB<N?TPB:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0);
    integrateGPUD<<<nblocks, nthreads>>>(pos, vel, force, step);
    //cudaCheckErrors("Integrate");  
  }

  struct dot_functor{
    __device__ real3 operator()(real3 &a){
      return a*a;
    }

  };

  real computeKineticEnergyGPU(real3 *vel, uint N){
    device_ptr<real3> d_vel3(vel);
    real3 K;
    thrust::plus<real3> binary_op;
  
    K = thrust::transform_reduce(d_vel3, d_vel3 + N,
				 dot_functor(), make_real3(0.0f), binary_op);
  
    return 0.5f*(K.x+K.y+K.z)/(real)N;
  }
}

/*Raul P. Pelaez 2016. Two step velocity VerletNVE Integrator GPU callers 

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
TODO:
100- K sum could be done better with some thrust magic
*/
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
  __global__ void integrateGPUD(float4 __restrict__  *pos,
				float3 __restrict__ *vel,
				const float4 __restrict__  *force,
				int step){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=params.N) return;

    /*Half step velocity*/
    vel[i] += make_float3(force[i])*params.dt*0.5f;
    
    if(params.L.z==0.0f) vel[i].z = 0.0f; //2D
    
    /*In the first step, upload positions*/
    if(step==1)
      pos[i] += make_float4(vel[i])*params.dt;
    
    
  }



  //Update the positions
  void integrateGPU(float4 *pos, float3 *vel, float4 *force, uint N, int step){
    
    uint nthreads = TPB<N?TPB:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0);
    integrateGPUD<<<nblocks, nthreads>>>(pos, vel, force, step);
    //cudaCheckErrors("Integrate");  
  }

  struct dot_functor{
    __device__ float3 operator()(float3 &a){
      return a*a;
    }

  };

  float computeKineticEnergyGPU(float3 *vel, uint N){
    device_ptr<float3> d_vel3(vel);
    float3 K;
    thrust::plus<float3> binary_op;
  
    K = thrust::transform_reduce(d_vel3, d_vel3 + N,
				 dot_functor(), make_float3(0.0f), binary_op);
  
    return 0.5f*(K.x+K.y+K.z)/(float)N;
  }
}

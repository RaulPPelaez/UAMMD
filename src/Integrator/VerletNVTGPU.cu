/*Raul P. Pelaez 2016. Two step velocity VerletNVT Integrator GPU callers 

  Functions to integrate movement.

  Uses a two step velocity verlet algorithm with a BBK thermostat.

  V[t+0.5dt] = V[t] -0.5·dt·(F[t]+gamma·V[t]) + sqrt(dt·0.5)·sigma·G[t]

  X[t+dt] = X[t] + dt·V[t+0.5·dt]

  V[t+dt] = V[t+dt·0.5] -0.5·dt·(F[t+dt]+gamma·V[t+dt]) + sqrt(dt·0.5)·sigma·G[t+dt]

TODO:
100- K sum could be done better with some thrust magic
*/
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include "VerletNVTGPU.cuh"
#include<iostream>
#include<thrust/device_ptr.h>
//#include<thrust/reduce.h>
#include<thrust/transform_reduce.h>

//Threads per block
#define TPB 128

/*All functions and variables are always englobed in a namespace*/
namespace verlet_nvt_ns{
  /*Parameters in constant memory*/
  __constant__ Params params;

  /*Initialize all necesary things on GPU*/
  void initVerletNVTGPU(Params m_params){
    /*Upload params to constant memory*/
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }


  /*Integrate the movement*/
  __global__ void integrateD(float4 *pos, float3 *vel, float4 *force, float3* noise,
			     uint N, int step){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    /*Half step velocity*/
    vel[i] += (make_float3(force[i])-params.gamma*vel[i])*params.dt*0.5f + params.noiseAmp*noise[i];

    /*In the first step, upload positions*/
    if(step==1)
      pos[i] += make_float4(vel[i])*params.dt;    
    
  }

  /*CPU kernel caller*/
  void integrateVerletNVTGPU(float4 *pos, float3 *vel, float4 *force, float3* noise,
			     uint N, int step){
    uint nthreads = TPB<N?TPB:N;
    uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0); 
    integrateD<<<nblocks, nthreads>>>(pos, vel, force, noise, N, step);
  }

  /*Returns the squared of each element in a float3*/
  struct dot_functor{
    __device__ float3 operator()(float3 &a){
      return a*a;
    }

  };

  /*Compute the kinetic energy from the velocities*/
  float computeKineticEnergyVerletNVT(float3 *vel, uint N){
    thrust::device_ptr<float3> d_vel3(vel);
    float3 K;
    thrust::plus<float3> binary_op;
  
    K = thrust::transform_reduce(d_vel3, d_vel3 + N,
				 dot_functor(), make_float3(0.0f), binary_op);

    //float3 Ptot = thrust::reduce(d_vel3, d_vel3+N, make_float3(0.0f));
    
    //std::cout<<Ptot.x<< " "<<Ptot.y<<" "<<Ptot.z<<std::endl;
    
    return 0.5f*(K.x+K.y+K.z)/(float)N;
  }
  
}

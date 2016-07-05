/*Raul P. Pelaez 2016. Two step velocity VerletNVT Integrator GPU callers 

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 

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
#include<thrust/device_ptr.h>
#include<thrust/reduce.h>
#include<thrust/transform_reduce.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>


__constant__ VNVTparams params;


void initVerletNVTGPU(VNVTparams m_params){
  gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(VNVTparams)));
}


using namespace thrust;
//This struct is a thrust trick to perform an arbitrary transformation
//In this case it performs a two step velocity verlet integration
//Performs a two step velocity verlet integrator, pick with step
struct twoStepVelVerletNVT_Integratefunctor{
  int step;
  __host__ __device__ twoStepVelVerletNVT_Integratefunctor(int step):
    step(step){}
  //The operation is performed on creation
  template <typename Tuple>
  __device__  void operator()(Tuple t){
    /*Retrive the data*/
    float4 pos = get<0>(t);
    float4 vel = make_float4(get<1>(t),0.0f);
    float4 force = get<2>(t);
    float4 noise = make_float4(get<3>(t));
    
    float dt = params.dt;
    float gamma = params.gamma;
    float noiseAmp = params.noiseAmp;
    
    switch(step){
      /*First velocity verlet step*/
    case 1: 
      vel += (force-gamma*vel)*dt*0.5f  + noiseAmp*noise; 
      vel.w = 0.0f; //Be careful not to overwrite the pos.w!!
      pos += vel*dt;
      get<0>(t) = pos;
      break;
      /*Second velocity verlet step*/
    case 2:
      vel += (force-gamma*vel)*dt*0.5f  + noiseAmp*noise; 
      break;
    }
    /*Write new vel*/
    get<1>(t) = make_float3(vel);
  }
};



//Update the positions
void integrateVerletNVTGPU(float4 *pos, float3 *vel, float4 *force, float3* noise,
			   uint N, int step){
  
  device_ptr<float4> d_pos4(pos);
  device_ptr<float3> d_vel3(vel);
  device_ptr<float3> d_noise3(noise);
  device_ptr<float4> d_force4(force);
  /**Thrust black magic to perform a triple transformation, see the functor description**/
  for_each(
	   make_zip_iterator( make_tuple( d_pos4, d_vel3, d_force4, d_noise3)),
	   make_zip_iterator( make_tuple( d_pos4 + N, d_vel3 + N, d_force4 + N, d_noise3 + N)),
	   twoStepVelVerletNVT_Integratefunctor(step));
  //cudaCheckErrors("Integrate");

  
}

struct dot_functor{
    __device__ float3 operator()(float3 &a){
    return a*a;
  }

};

float computeKineticEnergyVerletNVT(float3 *vel, uint N){
  device_ptr<float3> d_vel3(vel);
  float3 K;
  thrust::plus<float3> binary_op;
  
  K = thrust::transform_reduce(d_vel3, d_vel3 + N,
			       dot_functor(), make_float3(0.0f), binary_op);
  
  return 0.5f*(K.x+K.y+K.z)/(float)N;
}

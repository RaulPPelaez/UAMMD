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
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>


using namespace thrust;
//This struct is a thrust trick to perform an arbitrary transformation
//In this case it performs a two step velocity verlet integration
//Performs a two step velocity verlet integrator, pick with step
struct twoStepVelVerletNVE_Integratefunctor{
  float dt;
  int step;
  __host__ __device__ twoStepVelVerletNVE_Integratefunctor(float dt, int step):
    dt(dt),step(step){}
  //The operation is performed on creation
  template <typename Tuple>
  __device__  void operator()(Tuple t){
    /*Retrive the data*/
    float4 pos = get<0>(t);
    float4 vel = make_float4(get<1>(t),0.0f);
    float4 force = get<2>(t);
    switch(step){
      /*First velocity verlet step*/
    case 1: 
      vel += force*dt*0.5f;
      vel.w = 0.0f; //Be careful not to overwrite the pos.w!!
      pos += vel*dt;
      get<0>(t) = pos;
      break;
      /*Second velocity verlet step*/
    case 2:
      vel += force*dt*0.5f;
      break;
    }
    /*Write new vel*/
    get<1>(t) = make_float3(vel);
  }
};



//Update the positions
void integrateVerletNVEGPU(float4 *pos, float3 *vel, float4 *force,
				  float dt, uint N, int step){

  //Uncomment to sum total energy, you must compute it first in the interactors
   // static uint count = 0;
   // count++; 
  device_ptr<float4> d_pos4(pos);
  device_ptr<float3> d_vel3(vel);
  device_ptr<float4> d_force4(force);

  /**Thrust black magic to perform a triple transformation, see the functor description**/
  for_each(
	   make_zip_iterator( make_tuple( d_pos4, d_vel3, d_force4)),
	   make_zip_iterator( make_tuple( d_pos4 + N, d_vel3 + N, d_force4 +N)),
	   twoStepVelVerletNVE_Integratefunctor(dt, step));
  //cudaCheckErrors("Integrate");  
}

struct dot_functor{
    __device__ float3 operator()(float3 &a){
    return a*a;
  }

};

float computeKineticEnergyVerletNVE(float3 *vel, uint N){
  device_ptr<float3> d_vel3(vel);
  float3 K;
  thrust::plus<float3> binary_op;
  
  K = thrust::transform_reduce(d_vel3, d_vel3 + N,
			       dot_functor(), make_float3(0.0f), binary_op);
  
  return 0.5f*(K.x+K.y+K.z)/(float)N;
}

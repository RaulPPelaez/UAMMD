/*Raul P. Pelaez 2016. Integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include "TwoStepVelVerletGPU.cuh"
#include<thrust/device_ptr.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>


using namespace thrust;
//This struct is a thrust trick to perform an arbitrary transformation
//In this case it performs a two step velocity verlet integration
//Performs a two step velocity verlet integrator, pick with step
struct twoStepVelVerlet_functor{
  float dt;
  int step;
  bool dump;
  __host__ __device__ twoStepVelVerlet_functor(float dt, int step, bool dump):
    dt(dt),step(step), dump(dump){}
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
      pos += vel*dt;
      get<0>(t) = pos;
      //      get<2>(t) = make_float4(0.0f);
      break;
      /*Second velocity verlet step*/
    case 2:
      vel += force*dt*0.5f;
      if(dump) vel *= 0.99f;
      break;
    }
    /*Write new vel and reset force*/
    get<1>(t) = make_float3(vel);
  }
};

// struct brownianEulerMaruyama_functor{
//   float dt;
//   int step;
//   bool dump;
//   __host__ __device__ brownianEulerMaruyama_functor(float dt, int step, bool dump):
//     dt(dt),step(step), dump(dump){}
//   //The operation is performed on creation
//   template <typename Tuple>
//   __device__  void operator()(Tuple t){
//     /*Retrive the data*/
//     float4 pos = get<0>(t);
//     float4 dW = make_float4(get<1>(t),0.0f);
//     float4 force = get<2>(t);

//     fori(0,3){
//       pos.x +=  params.dt*(params.K[3*i+0]*pos.x + params.D[3*i+0]*force.x) + dW.x*params.B[3*i+0];    
//       pos.y +=  params.dt*(params.K[3*i+1]*pos.y + params.D[3*i+1]*force.y) + dW.y*params.B[3*i+1];    
//       pos.z +=  params.dt*(params.K[3*i+2]*pos.z + params.D[3*i+2]*force.z) + dW.z*params.B[3*i+2];
//     }
//   }
// };


//Update the positions
void integrateTwoStepVelVerletGPU(float4 *pos, float3 *vel, float4 *force, float dt, uint N, int step, bool dump){

  device_ptr<float4> d_pos4(pos);
  device_ptr<float3> d_vel3(vel);
  device_ptr<float4> d_force4(force);
  /**Thrust black magic to perform a triple transformation, see the functor description**/
  for_each(
	   make_zip_iterator( make_tuple( d_pos4, d_vel3, d_force4)),
	   make_zip_iterator( make_tuple( d_pos4 + N, d_vel3 + N, d_force4 +N)),
	   twoStepVelVerlet_functor(dt, step, dump));
  //cudaCheckErrors("Integrate");					   
}


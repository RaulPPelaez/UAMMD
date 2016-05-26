/*Raul P. Pelaez 2016. Integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
  
  Currently Implemented integrators:
    1. Velocity Verlet

TODO:
100- Template somehow the integrator functor to allow selection of integrator from Integrator class
100- Implement new integrators
*/
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include"IntegratorGPU.cuh"
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

//Update the positions
void integrate(float4 *pos, float3 *vel, float4 *force, float dt, uint N, int step, bool dump){

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


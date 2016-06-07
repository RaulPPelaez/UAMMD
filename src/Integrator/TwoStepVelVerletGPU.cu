/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator GPU callers 

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 

TODO:
100- Sum the energy in force.w only, not all the other 3 components.

*/
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include "TwoStepVelVerletGPU.cuh"
#include<thrust/device_ptr.h>
#include<thrust/reduce.h>
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
      pos.w = 0.0f;
      get<0>(t) = pos;
      break;
      /*Second velocity verlet step*/
    case 2:
      vel += force*dt*0.5f;
      if(dump) vel *= 0.99f;
      //Uncomment to sum kinetic energy
      // force.w += 0.5f*dot(vel,vel); 
      // get<2>(t).w = force.w;
      break;
    }
    /*Write new vel and reset force*/
    get<1>(t) = make_float3(vel);
  }
};

//Update the positions
void integrateTwoStepVelVerletGPU(float4 *pos, float3 *vel, float4 *force,
				  float dt, uint N, int step, bool dump){

  //Uncomment to sum total energy, you must compute it first in the interactors
  //This will print:  U K+U
  //So K = $2-$1
  // static uint count = 0;
  // count++; 
  device_ptr<float4> d_pos4(pos);
  device_ptr<float3> d_vel3(vel);
  device_ptr<float4> d_force4(force);

  /*This is super slow, you only need to sum force.w
  // float4 FUsum;
  // if(step==2 && count%1000==0){
  //   FUsum = thrust::reduce(d_force4, d_force4+N, make_float4(0.0f));
  //   std::cout<<(FUsum.w/(float)N)<<" ";
  // }  

  /**Thrust black magic to perform a triple transformation, see the functor description**/
  for_each(
	   make_zip_iterator( make_tuple( d_pos4, d_vel3, d_force4)),
	   make_zip_iterator( make_tuple( d_pos4 + N, d_vel3 + N, d_force4 +N)),
	   twoStepVelVerlet_functor(dt, step, dump));
  //cudaCheckErrors("Integrate");

  
  // if(step==2 && count%1000==0){
  //   FUsum = thrust::reduce(d_force4, d_force4+N, make_float4(0.0f));
  //   std::cout<<(FUsum.w/(float)N)<<std::endl;
  // }
  
}


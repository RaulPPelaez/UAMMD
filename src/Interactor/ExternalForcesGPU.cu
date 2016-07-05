
#include"ExternalForcesGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include<thrust/device_ptr.h>
#include<thrust/reduce.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>

using namespace thrust;

//Parameters in constant memory, super fast access
__constant__ ExternalForcesParams eFParamsGPU; 



void initExternalForcesGPU(ExternalForcesParams m_params){
  m_params.invL = 1.0f/m_params.L;
  /*Upload parameters to constant memory*/
  gpuErrchk(cudaMemcpyToSymbol(eFParamsGPU, &m_params, sizeof(ExternalForcesParams)));
}

//Thrust trick to apply a transformation to each element of an array, in parallel
struct externalForces_functor{
  __host__ __device__ externalForces_functor(){}
  //The operation is performed on creation
  template <typename Tuple>
  __device__  void operator()(Tuple t){
    /*Retrive the data*/
    float4 pos = get<0>(t);
    float4 force = get<1>(t);


    float L = 15.0f;
    float z = (pos.z+2.0f*L);
    float f;
    if(abs(z)<1.0f) f = 100000.0f*pow(1-z*z,4);
    else f=0.0f;
      
    force += make_float4(0.0f, 0.0f, f - 2.5f, 0.0f);

    
    get<1>(t) = force;
  }
};


void computeExternalForce(float4 *force, float4 *pos, uint N){

  device_ptr<float4> d_pos4(pos);
  device_ptr<float4> d_force4(force);

  /**Thrust black magic to perform a multiple transformation, see the functor description**/
  for_each(
	   make_zip_iterator( make_tuple( d_pos4, d_force4)),
	   make_zip_iterator( make_tuple( d_pos4 + N, d_force4 +N)),
	   externalForces_functor()); 

}


#include"globals/defines.h"
#include"ExternalForcesGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include<thrust/adjacent_difference.h>
#include<thrust/device_ptr.h>
#include<thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include<thrust/reduce.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>
#include<cuda_runtime.h>

using namespace thrust;


namespace external_forces_ns{

  //Parameters in constant memory, super fast access
  __constant__ Params params; 



  void initGPU(Params m_params){
    m_params.invL = 1.0/m_params.L;
    /*Upload parameters to constant memory*/
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }

  //Thrust trick to apply a transformation to each element of an array, in parallel
  struct externalForces_functor{
    __host__ __device__ externalForces_functor(){}
    //The operation is performed on creation
    template <typename Tuple>
    __device__  void operator()(Tuple t){
      /*Retrive the data*/
      real3 pos = make_real3(get<0>(t));
      real4 force = get<1>(t);

      get<1>(t) = force;
    }
  };


  void computeExternalForce(real4 *force, real4 *pos, uint N){
    
    device_ptr<real4> d_pos4(pos);
    device_ptr<real4> d_force4(force);

    for_each(
	     make_zip_iterator( make_tuple( d_pos4, d_force4)),
	     make_zip_iterator( make_tuple( d_pos4 + N, d_force4 +N)),
	     externalForces_functor()); 

  }
}


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
    real3 dir, cm;
    __host__ __device__ externalForces_functor(real3 dir, real3 cm): dir(dir), cm(cm){}
    //The operation is performed on creation
    template <typename Tuple>
    __device__  void operator()(Tuple t){
      /*Retrive the data*/
      real3 pos = make_real3(get<0>(t));
      real4 force = get<1>(t);

      real3 axis2pos = (cm-pos) - dot(cm-pos, dir)*dir;
      
      real3 fdir = normalize(cross(dir, axis2pos));

      force -= make_real4(fdir*dot(axis2pos, axis2pos));
    
      get<1>(t) = force;
    }
  };


  void computeExternalForce(real4 *force, real4 *pos, uint N, real4 *poscpu){
    
    device_ptr<real4> d_pos4(pos);
    device_ptr<real4> d_force4(force);

    static device_vector<real4> diffpos(N);

    real4 cm = thrust::reduce(d_pos4, d_pos4+N, make_real4(0.0));

    thrust::adjacent_difference(d_pos4, d_pos4+N, diffpos.begin());

    real4 dir = thrust::reduce(diffpos.begin(), diffpos.end(), make_real4(0.0));
    real3 dir3  = normalize(make_real3(dir));
    real3 cm3 = make_real3(cm/N);
    
    // std::cout<<cm3.x<<" "<<cm3.y<<" "<<cm3.z-100<<" 1 "<<dir3.x<<" "<<dir3.y<<" "<<dir3.z*100<<std::endl;
    // for(int i=0; i<N; i++){
    //   real3 axis2pos = (cm3-make_real3(poscpu[i])) - dot(cm3-make_real3(poscpu[i]), dir3)*dir3;
      
    //   real3 fdir = normalize(cross(dir3, axis2pos));
      
    //   std::cout<<poscpu[i].x<<" "<<poscpu[i].y<<" "<<poscpu[i].z<<" 0 "<<fdir.x<<" "<<fdir.y<<" "<<fdir.z<<std::endl;
    // }
    
    /**Thrust black magic to perform a multiple transformation, see the functor description**/
    for_each(
	     make_zip_iterator( make_tuple( d_pos4, d_force4)),
	     make_zip_iterator( make_tuple( d_pos4 + N, d_force4 +N)),
	     externalForces_functor(dir3, cm3)); 

  }
}

/*Raul P. Pelaez 2016. Brownian Euler Maruyama integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 

  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+D·F[t]) + sqrt(dt)·dW·B
   Being:
     X - Positions
     D - Diffusion matrix
     K - Shear matrix
     dW- Noise vector
     B - sqrt(D)


TODO:
100- Benchmark and optimize
*/
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include"BrownianEulerMaruyamaGPU.cuh"
#include<thrust/device_ptr.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>


using namespace thrust;

__constant__ BrownianEulerMaruyamaParameters BEMParamsGPU;


void initBrownianEulerMaruyamaGPU(BrownianEulerMaruyamaParameters m_params){

  gpuErrchk(cudaMemcpyToSymbol(BEMParamsGPU, &m_params, sizeof(BrownianEulerMaruyamaParameters)));
}

//This struct is a thrust trick to perform an arbitrary transformation
//In this case it performs a brownian euler maruyama integration
struct brownianEulerMaruyama_functor{
  __host__ __device__ brownianEulerMaruyama_functor(){}
  //The operation is performed on creation
  template <typename Tuple>
  __device__  void operator()(Tuple t){
    /*Retrive the data*/
    float4 pp = get<0>(t);
    float3 pos = make_float3(pp);
    float3 dW = get<1>(t);
    float3 force = make_float3(get<2>(t));

    float3 *B = BEMParamsGPU.B;
    float3 *D = BEMParamsGPU.D;
    float3 *K = BEMParamsGPU.K;
    float sqrtdt = BEMParamsGPU.sqrtdt;
    float dt = BEMParamsGPU.dt; 
    
    pos.x +=  dt*( dot(K[0],pos) +  dot(D[0],force)) + sqrtdt*dot(dW,B[0]);
    pos.y +=  dt*( dot(K[1],pos) +  dot(D[1],force)) + sqrtdt*dot(dW,B[1]);
    pos.z +=  dt*( dot(K[2],pos) +  dot(D[2],force)) + sqrtdt*dot(dW,B[2]);
    


    get<0>(t) = make_float4(pos, pp.w);
  }
};


//Update the positions
void integrateBrownianEulerMaruyamaGPU(float4 *pos, float3 *noise, float4 *force,
				       uint N){
  device_ptr<float4> d_pos4(pos);
  device_ptr<float3> d_noise3(noise);
  device_ptr<float4> d_force4(force);
  /**Thrust black magic to perform a triple transformation, see the functor description**/
  for_each(
	   make_zip_iterator( make_tuple( d_pos4, d_noise3, d_force4)),
	   make_zip_iterator( make_tuple( d_pos4 + N, d_noise3 + N, d_force4 +N)),
	   brownianEulerMaruyama_functor());
  //  cudaCheckErrors("Integrate");					   
}


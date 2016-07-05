
/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 

TODO:
100-Optimize, see .cpp
*/

#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include"BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<thrust/device_ptr.h>
#include<thrust/device_vector.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>


using namespace thrust;

__constant__ BrownianHydrodynamicsEulerMaruyamaParameters BEMParamsGPU;


void initBrownianHydrodynamicsEulerMaruyamaGPU(BrownianHydrodynamicsEulerMaruyamaParameters m_params){
  m_params.sqrt2dt = sqrt(2.0f)*m_params.sqrtdt;
  gpuErrchk(cudaMemcpyToSymbol(BEMParamsGPU, &m_params, sizeof(BrownianHydrodynamicsEulerMaruyamaParameters)));
}


struct copy_pos_functor{
  copy_pos_functor(){}
  __device__ float3 operator() (const float4& pos4){
    return make_float3(pos4);
  }
};

void copy_pos(float4 *pos, float3* pos3, float4 *force, float3* force3, uint N){
  
  device_ptr<float4> pos4(pos);
  device_ptr<float4> force4(force);
  
  device_ptr<float3> pos3d(pos3);
  device_ptr<float3> force3d(force3);
  
  transform(pos4, pos4+N, pos3d, copy_pos_functor());
  transform(force4, force4+N, force3d, copy_pos_functor());
  
}



//This struct is a thrust trick to perform an arbitrary transformation
//In this case it performs a brownian euler maruyama integration
struct brownianHydroEulerMaruyama_functor{
  float dt;
  __host__ __device__ brownianHydroEulerMaruyama_functor(float dt):
    dt(dt){}
  //The operation is performed on creation
  template <typename Tuple>
  __device__  void operator()(Tuple t){
    /*Retrive the data*/
    float4 pos = get<0>(t);
    float4 DF = make_float4(get<1>(t),0.0f);
    float4 BdW = make_float4(get<2>(t),0.0f);
    float4 KR = make_float4(get<3>(t),0.0f);
    
    pos += KR + DF + BEMParamsGPU.sqrt2dt*BdW;

    get<0>(t) = pos;
  }
};


//Update the positions
void integrateBrownianHydrodynamicsEulerMaruyamaGPU(float4 *pos, float3* DF, float3* BdW, float3* KR,
						    float dt, uint N){
  device_ptr<float4> d_pos4(pos);
  device_ptr<float3> d_DF3(DF);
  device_ptr<float3> d_BdW3(BdW);
  device_ptr<float3> d_KR3(KR);

  /**Thrust black magic to perform a general transformation, see the functor description**/
  for_each(
	   make_zip_iterator( make_tuple( d_pos4, d_DF3, d_BdW3, d_KR3)),
	   make_zip_iterator( make_tuple( d_pos4+N, d_DF3+N, d_BdW3+N, d_KR3+N)),
	   brownianHydroEulerMaruyama_functor(dt));
  //cudaCheckErrors("Integrate");					   
}





__global__ void fix_B(float *B, uint n){
  
  uint ii = blockIdx.x*blockDim.x + threadIdx.x;
  if(ii >= n*n) return;
  uint i = ii%n;
  uint j = ii/n;
  if(i<=j) return;  
  B[ii] = 0.0f;
}


void fix_B_call(float *B, uint N, cudaStream_t stream){
  fix_B<<<(3*N*3*N)/128+1,128,0, stream>>>(B, 3*N);
}







__global__ void rotneGPU(float *D, float3 *R2, uint N){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  
  int j = id/N;
  int i = id%N;
  float D0 = 1.0f;
    uint n = 3*N;
  if(i >= N || j>=N) return;
  else if(j==i){
    for(int k = 0; k < 3; k++)
      for(int l = 0; l < 3; l++){
	D[3*i + k + n*(3*i + l)] =  k==l?D0:0.0f;
      }
    return;
  }
  float rh = 1;
  float *R = (float*)R2;

  float rij[3];
  float r2 = 0.0f;
  float r;
  float c1, c2;
  for(int k = 0; k<3; k++){
    rij[k] = R[3*j + k] - R[3*i+k];
    r2 += rij[k]*rij[k];
  }
  r = sqrt(r2);
  if(r>=2*rh){
    c1 = 0.75*rh/r*(1.0f + 2.0f*rh*rh/(3.0f*r2));
    c2 = 0.75*rh/r*(1.0f - 2.0f*rh*rh/r2);
  }
  else{
    c1 = 1.0f - 9.0f*r/(32.0f*rh);
    c2 = 3.0f*r/(32.0f*rh);
  }

  for(int k = 0; k < 3; k++)
    for(int l = 0; l < 3; l++)
      D[3*i + k + n*(3*j + l)] = D0*c1*(k==l?1.0f:0.0f) + c2*rij[k]*rij[l]/r2;

}















__global__ void rotneGPU_prev(float *D, float3 *R, uint N){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  //int i = blockIdx.x;
  //int j = threadIdx.x;
  int j = id/N;
  int i = id%N;
  float D0 = 1.0f;
    uint n = 3*N;
  if(i >= N || j>=N ||  j<i) return;
  else if(j==i){
    for(int k = 0; k < 3; k++)
      for(int l = 0; l < 3; l++){
	D[3*i + k + n*(3*i + l)] =  0.0f;
      }
    D[3*i + 0 + n*(3*i + 0)] = D0;
    D[3*i + 1 + n*(3*i + 1)] = D0;
    D[3*i + 2 + n*(3*i + 2)] = D0;
    return;
  }
  float rh = 1;

  
  float3 rij;
  float r;
  float c1, c2;
  
  rij = R[j] - R[i];
  
  float invr2 = 1.0f/dot(rij, rij);

  
  float *rijp = &(rij.x);
  float invr = sqrt(invr2);
  
  r = 1.0f/invr;
  
  if(r>=2*rh){
    c1 = 0.75*rh*invr*(1.0f + 2.0f*invr2*rh*rh/3.0f);
    c2 = 0.75*rh*invr*(1.0f - 2.0f*rh*rh*invr2);
  }
  else{
    c1 = 1.0f - 9.0f*r/(32.0f*rh);
    c2 = 3.0f*r/(32.0f*rh);
  }

  
  for(int k = 0; k < 3; k++)
    for(int l = 0; l < 3; l++)
      D[3*i + k + n*(3*j + l)] = c2*rijp[k]*rijp[l]*invr2;

  for(int k = 0; k<3; k++)  D[3*i + k + n*(3*j + k)] += D0*c1; 
}


void rodne_call(float *d_D, float3 *d_R, cudaStream_t stream, uint N){
  rotneGPU_prev<<<N*N/128+1, 128, 0 ,stream>>>(d_D, d_R, N);
}








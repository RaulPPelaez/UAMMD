/*Raul P. Pelaez 2016. Integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"
#include"BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<thrust/device_ptr.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>


using namespace thrust;

__constant__ BrownianHydrodynamicsEulerMaruyamaParameters BEMParamsGPU;


void initBrownianHydrodynamicsEulerMaruyamaGPU(BrownianHydrodynamicsEulerMaruyamaParameters m_params){

  gpuErrchk(cudaMemcpyToSymbol(BEMParamsGPU, &m_params, sizeof(BrownianHydrodynamicsEulerMaruyamaParameters)));
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

    pos += KR + DF + BdW;

    get<0>(t) = pos;
  }
};


//Update the positions
void integrateBrownianHydrodynamicsEulerMaruyamaGPU(float4 *pos, float3* DF, float3* BdW, float3* KR,
						    float dt, uint N);
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






















__global__ void fix_B(float *B){
  int ii = blockIdx.x*blockDim.x + threadIdx.x;
  if(ii>=n*n) return;
  int i = ii%n;
  int j = ii/n;
  if(i<=j) return;  
  B[ii] = 0.0f;
}

void rodne_call(float *d_D, float *d_R, cudaStream_t stream){
  rotneGPU<<<(particles*particles)/TPB+1, TPB, 0 ,stream>>>(d_D, d_R);
  //rotneGPU<<<particles, particles>>>(d_D, d_R);
}

void fix_B_call(float *B){
  fix_B<<<(n*n)/TPB+1,TPB>>>(B);
}


__global__ void rotneGPU(float *D, float *R){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  //int i = blockIdx.x;
  //int j = threadIdx.x;
  int j = id/particles;
  int i = id%particles;
  if(i>particles || j>particles || j==i) return;
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










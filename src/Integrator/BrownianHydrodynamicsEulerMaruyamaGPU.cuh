/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/

#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH

#include<thrust/device_vector.h>
namespace brownian_hy_euler_maruyama_ns{
  struct Params{
    float sqrtdt;
    float dt;
    float sqrt2dt;
    float4* K;
    float D0, rh;
    float3 L;
    uint N;
  };


  void float4_to_float3GPU(float4 *force, float3* force3, uint N);

  void rodne_callGPU(float *d_D, float4 *d_R, cudaStream_t stream, uint N);


  void initGPU(Params m_params);

  void integrateGPU(float4 *pos, float3* DF, float3* BdW, float4 *K, uint N);
}
#endif

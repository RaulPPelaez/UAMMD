/*Raul P. Pelaez 2016. Brownian Euler Maruyama integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef INTEGRATORBROWNIANEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANEULERMARUYAMAGPU_CUH


namespace brownian_euler_maruyama_ns{
  struct Params{
    float sqrtdt, dt;
    float3 *B, *D, *K;
    float3 L;
    uint N;
  };

  void initGPU(Params m_params);

  void integrateGPU(float4 *pos, float3 *noise, float4 *force, uint N);
}
#endif

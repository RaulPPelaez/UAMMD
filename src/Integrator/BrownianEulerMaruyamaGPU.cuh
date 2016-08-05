/*Raul P. Pelaez 2016. Brownian Euler Maruyama integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef INTEGRATORBROWNIANEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANEULERMARUYAMAGPU_CUH


struct BrownianEulerMaruyamaParameters{
  float sqrtdt, dt;
  float3 *B, *D, *K;
};

void initBrownianEulerMaruyamaGPU(BrownianEulerMaruyamaParameters m_params);

void integrateBrownianEulerMaruyamaGPU(float4 *pos, float3 *vel, float4 *force,
				       uint N);

#endif

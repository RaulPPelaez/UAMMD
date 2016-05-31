/*Raul P. Pelaez 2016. Integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
  
  Currently Implemented integrators:
    1. Velocity Verlet
*/
#ifndef INTEGRATORBROWNIANEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANEULERMARUYAMAGPU_CUH


struct BrownianEulerMaruyamaParameters{
  float sqrtdt;
  float4 *B, *D, *K;
};

void initBrownianEulerMaruyamaGPU(BrownianEulerMaruyamaParameters m_params);

void integrateBrownianEulerMaruyamaGPU(float4 *pos, float3 *vel, float4 *force,
				       float dt, uint N);

#endif

/*Raul P. Pelaez 2016. Integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
  
  Currently Implemented integrators:
    1. Velocity Verlet
*/
#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH


struct BrownianHydrodynamicsEulerMaruyamaParameters{
  float sqrtdt;
  float4 *B, *D, *K;
};



void rodne_call(float *d_D, float *d_R, cudaStream_t stream);

void fix_B_call(float *B);


void initBrownianHydrodynamicsEulerMaruyamaGPU(BrownianHydrodynamicsEulerMaruyamaParameters m_params);

void integrateBrownianHydrodynamicsEulerMaruyamaGPU(float4 *pos, float3* DF, float3* BdW, float3* KR,
						    float dt, uint N);

#endif

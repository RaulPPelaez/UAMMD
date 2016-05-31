/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator GPU callers 

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef INTEGRATORTWOSTEPVELVERLETGPU_CUH
#define INTEGRATORTWOSTEPVELVERLETGPU_CUH

void integrateTwoStepVelVerletGPU(float4 *pos, float3 *vel, float4 *force, float dt, uint N, int step, bool dump=false);

#endif

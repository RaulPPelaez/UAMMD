/*Raul P. Pelaez 2016. Integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
  
  Currently Implemented integrators:
    1. Velocity Verlet
*/
#ifndef INTEGRATORTWOSTEPVELVERLETGPU_CUH
#define INTEGRATORTWOSTEPVELVERLETGPU_CUH

void integrateTwoStepVelVerletGPU(float4 *pos, float3 *vel, float4 *force, float dt, uint N, int step, bool dump=false);

#endif

/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator GPU callers 

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef VERLETNVEGPU_CUH
#define VERLETNVEGPU_CUH

void integrateVerletNVEGPU(float4 *pos, float3 *vel, float4 *force, float dt, uint N, int step);


float computeKineticEnergyVerletNVE(float3 *vel, uint N);

#endif

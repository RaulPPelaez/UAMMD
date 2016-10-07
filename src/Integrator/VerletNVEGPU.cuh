/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator GPU callers 

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef VERLETNVEGPU_CUH
#define VERLETNVEGPU_CUH

namespace verlet_nve_ns{
  
  struct Params{
    float dt;
    uint N;
    float3 L;
  };

  void initGPU(Params params);
  void integrateGPU(float4 *pos, float3 *vel, float4 *force, uint N, int step);
  

  float computeKineticEnergyGPU(float3 *vel, uint N);

}
#endif

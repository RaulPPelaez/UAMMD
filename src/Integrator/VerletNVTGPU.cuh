/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator GPU callers  NVT

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef VERLETNVTGPU_CUH
#define VERLETNVTGPU_CUH

/*Each module should have its own namespace*/
namespace verlet_nvt_ns{

  struct Params{
    float dt;
    float gamma;
    float T;
    float noiseAmp;
    float3 L;
    uint N;
  };

  void initGPU(Params m_params);

  /*Step controls the integration step, 1 or 2 in verlet*/
  void integrateGPU(float4 *pos, float3 *vel, float4 *force, float3 *noise, uint N,
		    int step);

  float computeKineticEnergyGPU(float3 *vel, uint N);
}
#endif

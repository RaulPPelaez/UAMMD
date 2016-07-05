/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator GPU callers  NVT

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef VERLETNVTGPU_CUH
#define VERLETNVTGPU_CUH



struct VNVTparams{
  float dt;
  float gamma;
  float T;
  float noiseAmp;
};

void initVerletNVTGPU(VNVTparams m_params);


void integrateVerletNVTGPU(float4 *pos, float3 *vel, float4 *force, float3 *noise,
			   uint N, int step);


float computeKineticEnergyVerletNVT(float3 *vel, uint N);

#endif

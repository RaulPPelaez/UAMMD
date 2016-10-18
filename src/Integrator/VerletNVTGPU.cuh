/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator GPU callers  NVT

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef VERLETNVTGPU_CUH
#define VERLETNVTGPU_CUH
#include"globals/defines.h"
/*Each module should have its own namespace*/
namespace verlet_nvt_ns{

  struct Params{
    real dt;
    real gamma;
    real T;
    real noiseAmp;
    real3 L;
    uint N;
  };

  void initGPU(Params m_params);

  /*Step controls the integration step, 1 or 2 in verlet*/
  void integrateGPU(real4 *pos, real3 *vel, real4 *force, real3 *noise, uint N,
		    int step);

  real computeKineticEnergyGPU(real3 *vel, uint N);
}
#endif

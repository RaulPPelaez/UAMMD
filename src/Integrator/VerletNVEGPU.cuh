/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator GPU callers 

  Functions to integrate movement. The integration is done via a functor wich creator
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef VERLETNVEGPU_CUH
#define VERLETNVEGPU_CUH
#include "globals/defines.h"
namespace verlet_nve_ns{
  
  struct Params{
    real dt;
    uint N;
    real3 L;
  };

  void initGPU(Params params);
  void integrateGPU(real4 *pos, real4 *pos1, real3 *vel, real4 *force, uint N, int step);
  

  real computeKineticEnergyGPU(real3 *vel, uint N);

}
#endif

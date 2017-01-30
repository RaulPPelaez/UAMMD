//WARNING: DIFFUSION REFERS TO MOBILITY M = D/kT
/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics integrator GPU kernels and callers
*/

#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH

#include<thrust/device_vector.h>
#include"globals/defines.h"
namespace brownian_hy_euler_maruyama_ns{
  struct Params{
    real dt;
    real sqrt2Tdt;
    real invDelta;
    real T;
    real3 L;
    uint N;
  };

  void real4_to_real3GPU(real4 *force, real3* force3, uint N);


  void initGPU(Params m_params);
  
  void integrateGPU(real4 *pos, real3* MF, real3* BdW,  real3* divM, real3 *K, uint N);

}
#endif

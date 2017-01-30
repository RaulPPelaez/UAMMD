/*Raul P. Pelaez 2016. Brownian Euler Maruyama integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/
#ifndef INTEGRATORBROWNIANEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANEULERMARUYAMAGPU_CUH
#include"globals/defines.h"

namespace brownian_euler_maruyama_ns{
  struct Params{
    real sqrt2Tdt, dt;
    real3 *B, *D, *K;
    real3 L;
    uint N;
  };

  void initGPU(Params m_params);

  void integrateGPU(real4 *pos, real3 *noise, real4 *force, uint N);
}
#endif

/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics integrator GPU kernels and callers

  Functions to integrate movement. The integration is done via a functor wich accesor ()
     takes a thrust::Tuple containing positions, velocities and forces on each particle. 
*/

#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH

#include<thrust/device_vector.h>
#include"globals/defines.h"
namespace brownian_hy_euler_maruyama_ns{
  struct Params{
    real sqrtdt;
    real dt;
    real sqrt2dt;
    real3 L;
    uint N;
  };
  /*Rodne-Prager-Yamakawa parameters*/
  struct RPYParams{
    real D0, rh;
    real inv32rhtimes3;
    real rhrh2div3;

  };

  void real4_to_real3GPU(real4 *force, real3* force3, uint N);

  void computeDiffusionRPYGPU(real *d_D, real4 *d_R, cudaStream_t stream, uint N);

  void diffusionDotGPU(real4 *pos, real3 *v, real3 *Dv, uint N);

  void initGPU(Params m_params);

  void initRPYGPU(RPYParams m_params);
  
  void integrateGPU(real4 *pos, real3* DF, real3* BdW, real3 *K, uint N);

}
#endif

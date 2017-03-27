/* Raul P. Pelaez 2016. GPU defines for the Mobility handler*/
#ifndef DIFFUSIONBDHIGPU_CUH
#define DIFFUSIONBDHIGPU_CUH

namespace brownian_hy_euler_maruyama_ns{

  /*Rodne-Prager-Yamakawa parameters*/
  struct RPYParams{
    real D0, rh;
    real inv32rhtimes3;
    real rhrh2div3;

  };

  /*Fills a 3Nx3N array with the mobility matrix*/
  void computeDiffusionRPYGPU(real *d_D, real4 *d_R, cudaStream_t stream, uint N);

  /*Computes the product M·v, needs the positions and v, retuns M·F in Dv*/
  void diffusionDotGPU(real4 *pos, real3 *v, real3 *Dv, uint N, cudaStream_t st = 0, bool divergence_mode = false);
  /*Upload parameters to GPU*/
  void initRPYGPU(RPYParams m_params);
  /*Computes the divergence term in 2D from the analytic formula, as an Nbody force*/
  void divergenceGPU(real4 *pos, real3 *divM, uint N);

}
#endif
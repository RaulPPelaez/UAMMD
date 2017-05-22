/*Raul P. Pelaez 2017. BDHI Lanczos submodule.
  
  Computes the mobility matrix on the fly when needed, so it is a mtrix free method.

  M·F is computed as an NBody interaction (a dense Matrix vector product).

  BdW is computed using the Lanczos algorithm [1].

References:
[1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations.
         -http://dx.doi.org/10.1063/1.4742347
*/
#ifndef BDHI_LANCZOS_CUH
#define BDHI_LANCZOS_CUH

#include "BDHI.cuh"
#include "misc/LanczosAlgorithm.cuh"
namespace BDHI{
  class Lanczos: public BDHI_Method{
  public:
    Lanczos(real M0, real rh, int N, real tolerance = 1e-3);
    ~Lanczos();
    void setup_step(              cudaStream_t st = 0) override{};
    void computeMF(real3* MF,     cudaStream_t st = 0) override;    
    void computeBdW(real3* BdW,   cudaStream_t st = 0) override;  
    void computeDivM(real3* divM, cudaStream_t st = 0) override;
    
    
  private:
    /*Kernel launch parameters*/
    int Nthreads, Nblocks;
    
    /*Rodne Prager Yamakawa device functions and parameters*/
    BDHI::RPYUtils utilsRPY;
    
    LanczosAlgorithm lanczosAlgorithm;
  };
}
#endif

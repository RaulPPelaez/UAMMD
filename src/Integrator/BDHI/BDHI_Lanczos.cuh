/*Raul P. Pelaez 2017. BDHI Lanczos submodule. Intended to be used with BDHI::EulerMaruyama
  
  Computes the mobility matrix on the fly when needed, so it is a mtrix free method.

  M·F is computed as an NBody interaction (a dense Matrix vector product).

  BdW is computed using the Lanczos algorithm [1].

  divM is computed as an NBody interaction

References:
[1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations.
         -http://dx.doi.org/10.1063/1.4742347
*/
#ifndef BDHI_LANCZOS_CUH
#define BDHI_LANCZOS_CUH

#include "BDHI.cuh"
#include "misc/LanczosAlgorithm.cuh"
namespace uammd{
  namespace BDHI{
    class Lanczos{
    public:
      using Parameters = BDHI::Parameters;
      Lanczos(shared_ptr<ParticleData> pd,
	      shared_ptr<ParticleGroup> pg,
	      shared_ptr<System> sys,
	      Parameters par);
      ~Lanczos();
      void setup_step(              cudaStream_t st = 0){};
      void computeMF(real3* MF,     cudaStream_t st = 0);    
      void computeBdW(real3* BdW,   cudaStream_t st = 0);  
      void computeDivM(real3* divM, cudaStream_t st = 0);
      void finish_step(cudaStream_t st = 0){};
    
    
    private:
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
      shared_ptr<System> sys;
       
      /*Rodne Prager Yamakawa device functions and parameters*/
      BDHI::RotnePragerYamakawa rpy;
      
      shared_ptr<LanczosAlgorithm> lanczosAlgorithm;

      curandGenerator_t curng;
      real hydrodynamicRadius;
      real temperature;
      real tolerance;
    };
  }
}
#include"BDHI_Lanczos.cu"
#endif

/*Raul P. Pelaez 2017-2022. BDHI Lanczos submodule. Intended to be used with BDHI::EulerMaruyama

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

#include"uammd.cuh"
#include "BDHI.cuh"
#include "misc/LanczosAlgorithm.cuh"
namespace uammd{
  namespace BDHI{
    class Lanczos{
    public:
      using Parameters = BDHI::Parameters;
      Lanczos(shared_ptr<ParticleData> pd, Parameters par):
	Lanczos(std::make_shared<ParticleGroup>(pd, "All"), par){}

      Lanczos(shared_ptr<ParticleGroup> pg, Parameters par);

      ~Lanczos(){}

      void setup_step(              cudaStream_t st = 0){};
      void computeMF(real3* MF,     cudaStream_t st = 0);
      void computeBdW(real3* BdW,   cudaStream_t st = 0);
      void computeDivM(real3* divM, cudaStream_t st = 0);
      void finish_step(cudaStream_t st = 0){};

      real getHydrodynamicRadius(){
	return par.hydrodynamicRadius;
      }
      real getSelfMobility(){
	long double rh = par.hydrodynamicRadius;
	if(rh<0) return -1.0;
	else return  1.0l/(6.0l*M_PIl*par.viscosity*rh);
      }


    private:
      shared_ptr<ParticleGroup> pg;

      /*Rodne Prager Yamakawa device functions and parameters*/
      BDHI::RotnePragerYamakawa rpy;

      std::shared_ptr<lanczos::Solver> lanczosAlgorithm;

      curandGenerator_t curng;
      real hydrodynamicRadius;
      real temperature;
      real tolerance;
      Parameters par;
    };
  }
}
#include"BDHI_Lanczos.cu"
#endif

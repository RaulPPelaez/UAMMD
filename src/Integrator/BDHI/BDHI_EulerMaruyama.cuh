/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation
  
  Solves the following stochastic differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*k*T)·B·dW + div(M)
   Being:
     X - Positions
     M - Mobility matrix
     K - Shear matrix
     dW- A collection of independent standard Wiener processes
     B - B*B^T = M -> i.e Cholesky decomposition B=chol(M) or Square root B=sqrt(M)
     div(M) - Divergence of the mobility matrix, only non-zero in 2D

 The Diffusion matrix is computed via the Rotne Prager Yamakawa tensor

 The module offers several ways to compute and sovle the different terms.

 The brownian Noise can be computed by:
     -Computing B·dW explicitly performing a Cholesky decomposition on M.
     -Through a Lanczos iterative method to reduce M to a smaller Krylov subspace and performing the operation there.

  On the other hand the mobility(diffusion) matrix can be handled in several ways:
     -Storing and computing it explicitly as a 3Nx3N matrix.
     -Not storing it and recomputing it when a product M·v is needed.

REFERENCES:

1- Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
        J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347

*/

#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_CUH
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_CUH
#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"global/defines.h"

#include"Integrator/Integrator.cuh"

#include"BDHI.cuh"
#include<curand.h>

namespace uammd{
  namespace BDHI{
  /*-----------------------------INTEGRATOR CLASS----------------------------------*/
  template<class Method>
  class EulerMaruyama: public Integrator{
  public:
    using Parameters = BDHI::Parameters;
    EulerMaruyama(shared_ptr<ParticleData> pd,
		  shared_ptr<ParticleGroup> pg,
		  shared_ptr<System> sys,		       
		  Parameters par);
    EulerMaruyama(shared_ptr<ParticleData> pd,
		  shared_ptr<System> sys,		       
		  Parameters par):
      EulerMaruyama(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), par){}
      
				     
    ~EulerMaruyama();

    void forwardTime() override;
    real sumEnergy() override;
  
  private:  
  
    thrust::device_vector<real3> MF;  /*Result of M·F*/
    thrust::device_vector<real3> BdW;  /*Result of B·dW*/
    thrust::device_vector<real3> divM;/*Divergence of the mobility Matrix, only in 2D*/
    thrust::device_vector<real3> K; /*Shear 3x3 matrix*/
    

    cudaStream_t stream, stream2;
    /*The method for computing the hydrodynamic interactions.
      Mainly in charge of computing MF, BdW and divM*/
    shared_ptr<Method> bdhi;

    Parameters par;

    int steps;
  };

  }
}

#include "BDHI_EulerMaruyama.cu"
#endif


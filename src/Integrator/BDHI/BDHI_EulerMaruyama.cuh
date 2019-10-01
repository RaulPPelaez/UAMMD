/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation

  Solves the following stochastic differential equation:
     X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*kb*T*dt)·B·dW + T·divM·dt(in 2D)
   Being:
     X - Positions
     M - Mobility matrix
     K - Shear matrix
     dW- A collection of independent standard Wiener processes
     B - B*B^T = M -> i.e Cholesky decomposition B=chol(M) or Square root B=sqrt(M)
     div(M) - Divergence of the mobility matrix, only non-zero in 2D

  The Mobility matrix is computed via the Rotne Prager Yamakawa tensor.

  The module offers several ways to compute and solve the different terms.

  BDHI::Cholesky:
  -Computing M·F and B·dW  explicitly storing M and performing a Cholesky decomposition on M.

  BDHI::Lanczos:
  -A Lanczos iterative method to reduce M to a smaller Krylov subspace and performing the operation B·dW there, the product M·F is performed in a matrix-free way, recomputing M every time M·v is needed.

  BDHI::PSE:
  -The Positively Split Edwald Method, which takes the computation to fourier space. [2]

  BDHI::FCM:
  -Fluctuating Froce Coupling Method, also spectral but computing only long range [3]

REFERENCES:


  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
    J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347
  [2] Rapid sampling of stochastic displacements in Brownian dynamics simulations
    The Journal of Chemical Physics 146, 124116 (2017); doi: http://dx.doi.org/10.1063/1.4978242
  [3] Fluctuating force-coupling method for simulations of colloidal suspensions;
    J. Comp. Phys. 269 (2014); doi: https://doi.org/10.1016/j.jcp.2014.03.013

*/

#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_CUH
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_CUH
#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"global/defines.h"

#include"Integrator/Integrator.cuh"
#include"utils/cxx_utils.h"
#include"Integrator/BDHI/BDHI.cuh"
#include<curand.h>

namespace uammd{
  namespace BDHI{
  /*-----------------------------INTEGRATOR CLASS----------------------------------*/
  template<class Method>
  class EulerMaruyama: public Integrator{
  public:
    using Parameters = typename Method::Parameters;
    EulerMaruyama(shared_ptr<ParticleData> pd,
		  shared_ptr<ParticleGroup> pg,
		  shared_ptr<System> sys,
		  Parameters par);
    EulerMaruyama(shared_ptr<ParticleData> pd,
		  shared_ptr<System> sys,
		  Parameters par):
      EulerMaruyama(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, par){}


    ~EulerMaruyama();

    void forwardTime() override;
    real sumEnergy() override;




    real getHydrodynamicRadius(){
      return bdhi->getHydrodynamicRadius();
    }

    real getSelfMobility(){
      return bdhi->getSelfMobility();
    }

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


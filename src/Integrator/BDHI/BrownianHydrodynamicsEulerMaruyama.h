/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation
   
  Solves the following stochastic differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*k*T)·B·dW
   Being:
     X - Positions
     M - Mobility matrix
     K - Shear matrix
     dW- A collection of independent standard Wiener processes
     B - B*B^T = M -> i.e Cholesky decomposition B=chol(M) or Square root B=sqrt(M)

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

#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_H
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_H
#include"globals/defines.h"
#include"globals/globals.h"
#include"utils/utils.h"
#include"Integrator.h"
#include"BDHI.cuh"
#include"BDHI_Cholesky.cuh"
#include"BDHI_Lanczos.cuh"
#include"BDHI_PSE.cuh"

enum BDHIMethod {CHOLESKY, LANCZOS, PSE, DEFAULT};

/*-----------------------------INTEGRATOR CLASS----------------------------------*/
class BrownianHydrodynamicsEulerMaruyama: public Integrator{
public:
  
  BrownianHydrodynamicsEulerMaruyama(Matrixf K, real vis, real rh,
				     BDHIMethod BDHIMethod = DEFAULT,
				     int max_iter = 0);
				     
  ~BrownianHydrodynamicsEulerMaruyama();

  void update() override;
  real sumEnergy() override;
  
private:  
  
  Vector3 MF;  /*Result of M·F*/
  GPUVector3 BdW;  /*Result of B·dW*/
  GPUVector3 divM;/*Divergence of the mobility Matrix, only in 2D*/
  Matrixf K; /*Shear 3x3 matrices*/

  cudaStream_t stream, stream2;
  /*The method for computing the hydrodynamic interactions.
    Mainly in charge of computing MF, BdW and divM*/
  shared_ptr<BDHI::BDHI_Method> bdhi;

  int nblocks, nthreads;
};



#endif

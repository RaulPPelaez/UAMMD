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
#include "utils/utils.h"
#include "Integrator.h"
#include "BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<curand.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<cuda_runtime.h>
#include"utils/cuda_lib_defines.h"

/*How the diffusion matrix will be handled,
  -MATRIXFULL stores a 3Nx3N matrix, computes it once and performs a matrix vector multiplication when needed
  -MATRIXFREE doesnt store a D matrix, and recomputes it on the fly when asked to multiply it by a vector
  -DEFAULT is FULL

*/
enum DiffusionMatrixMode{DEFAULT,MATRIXFULL, MATRIXFREE};
/*Method of obtaining the Brownian noise vector y = sqrt(D)·z
 -Cholesky Performs a Choesky decomposition on D and explicitly multiplies it by z, needs FULL matrix mode.
 -LANCZOS Performs a Krylov subspace reduction on D, and computes y in a much smaller subspace.
 */
enum StochasticNoiseMethod{CHOLESKY, LANCZOS};

#include "DiffusionBDHI.h"
#include "BrownianNoiseBDHI.h"
/*-----------------------------INTEGRATOR CLASS----------------------------------*/
class BrownianHydrodynamicsEulerMaruyama: public Integrator{
public:
  BrownianHydrodynamicsEulerMaruyama(Matrixf D0, Matrixf K,
				     StochasticNoiseMethod stochMethod = CHOLESKY,
				     DiffusionMatrixMode mode=DEFAULT, int max_iter = 0);
				     
  ~BrownianHydrodynamicsEulerMaruyama();

  void update() override;
  real sumEnergy() override;
  
private:

  Vector3 force3; /*Cublas needs a real3 array instead of real4 to multiply matrices*/
  
  Vector3 DF;  /*Result of D·F*/
  Vector3 divM;/*Divergence of the mobility Matrix, only in 2D*/
  Matrixf K, D0; /*Shear and self mobility 3x3 matrices*/

  brownian_hy_euler_maruyama_ns::Params params; /*GPU parameters (CPU version)*/
  
  cudaStream_t stream, stream2;

  /*Mobility handler*/
  shared_ptr<brownian_hy_euler_maruyama_ns::Diffusion> D; 
  /*Brownian noise computer, a shared pointer to the virtual base class*/
  shared_ptr<brownian_hy_euler_maruyama_ns::BrownianNoiseComputer> cuBNoise;
  
  cublasStatus_t status;
  cublasHandle_t handle;
};



#endif

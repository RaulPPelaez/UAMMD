/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation
   
  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+D·F[t]) + sqrt(dt)·dW·B
   Being:
     X - Positions
     D - Diffusion matrix
     K - Shear matrix
     dW- Noise vector
     B - sqrt(D)

  Similar to Brownian Euler Maruyama, but now the Diffusion matrix has size 3Nx3N and is updated
    each step according to the Rotne Prager method.
*/


#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_H
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMA_H
#include "utils/utils.h"
#include "Integrator.h"
#include "BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<curand.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<cuda_runtime.h>

class BrownianHydrodynamicsEulerMaruyama: public Integrator{
public:
  BrownianHydrodynamicsEulerMaruyama();
				     
  ~BrownianHydrodynamicsEulerMaruyama();

  void update() override;
  float sumEnergy() override;
  
private:

  Vector3 force3;
  
  void chol();
  
  Vector3 noise, DF;
  Matrixf D, K;
  
  curandGenerator_t rng;
  brownian_hy_euler_maruyama_ns::Params params;
  
  cudaStream_t stream, stream2;

  cusolverDnHandle_t solver_handle;

  cublasStatus_t status;
  cublasHandle_t handle;
};



#endif

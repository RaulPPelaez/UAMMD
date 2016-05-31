/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
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


#ifndef INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH
#define INTEGRATORBROWNIANHYDRODYNAMICSEULERMARUYAMAGPU_CUH
#include "utils/utils.h"
#include "Integrator.h"
#include "BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<curand.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<thread>



class BrownianHydrodynamicsEulerMaruyama: public Integrator{
public:
  BrownianHydrodynamicsEulerMaruyama(shared_ptr<Vector<float4>> pos,
			shared_ptr<Vector<float4>> force,
			shared_ptr<Vector<float4>> D,
			shared_ptr<Vector<float4>> K,
			uint N, float L, float dt);
  ~BrownianHydrodynamicsEulerMaruyama();

  void update() override;

private:

  void rodne_prage(cudaStream_t stream);
  void chol(cudaStream_t stream);
  void MVprod(float *M, float *V, float *resV, uint N, float alpha, float beta);
  Vector<float3> noise, KR, DF, BdW;
  curandGenerator_t rng;
  BrownianHydrodynamicsEulerMaruyamaParameters params;
  Matrix<float> D, B, K;
  cudaStream_t stream, stream2;

  cusolverDnHandle_t solver_handle;

  cublasStatus_t status;
  cublasHandle_t cublas_handle;

};



#endif

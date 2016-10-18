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
#include"globals/defines.h"
#include "utils/utils.h"
#include "Integrator.h"
#include "BrownianHydrodynamicsEulerMaruyamaGPU.cuh"
#include<curand.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<cuda_runtime.h>


#if defined SINGLE_PRECISION
#define cusolverDnpotrf cusolverDnSpotrf
#define cusolverDnpotrf_bufferSize cusolverDnSpotrf_bufferSize
#define cublastrmv cublasStrmv
#define cublassymv cublasSsymv
#else
#define cusolverDnpotrf cusolverDnDpotrf
#define cusolverDnpotrf_bufferSize cusolverDnDpotrf_bufferSize
#define cublastrmv cublasDtrmv
#define curandGenerateNormal curandGenerateNormalDouble
#define cublassymv cublasDsymv
#endif

struct cuSolverCholHandle{

  bool init(real *D, uint N){
    cusolverDnCreate(&solver_handle);
    h_work_size = 0;//work size of operation
    
    cusolverDnpotrf_bufferSize(solver_handle, 
				CUBLAS_FILL_MODE_UPPER, 3*N, D, 3*N, &h_work_size);
    gpuErrchk(cudaMalloc(&d_work, h_work_size*sizeof(real)));
    gpuErrchk(cudaMalloc(&d_info, sizeof(int)));
    this->D = D;
    this->N = N;
    
    return true;
  }

  bool compute(real *Dext=nullptr, uint Next=0, cudaStream_t stream = 0){
    real *Dcomp = Dext;
    uint Ncomp = Next;
    if(!Dext) Dcomp = this->D;
    if(!Next) Ncomp = this->N;

    cusolverDnSetStream(solver_handle, stream);
    cusolverDnpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		      3*Ncomp, Dcomp, 3*Ncomp, d_work, h_work_size, d_info);
    // int m_info;
    // cudaMemcpy(&m_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    // cerr<<" "<<m_info<<endl;
    return true;
  }
  
  real *D;
  uint N;
  cusolverDnHandle_t solver_handle;
  int h_work_size;
  real *d_work;
  int *d_info;

  
  
};

class BrownianHydrodynamicsEulerMaruyama: public Integrator{
public:
  BrownianHydrodynamicsEulerMaruyama(Matrixf D0, Matrixf K);
				     
  ~BrownianHydrodynamicsEulerMaruyama();

  void update() override;
  real sumEnergy() override;
  
private:

  Vector3 force3; /*Cublas needs a real3 array instead of real4 to multiply matrices*/
  
  Vector3 noise, DF;
  Vector<real> BdW;
  Matrixf D, K, D0;
  
  curandGenerator_t rng;
  brownian_hy_euler_maruyama_ns::Params params;
  
  cudaStream_t stream, stream2;

  cuSolverCholHandle cuChol;

  cublasStatus_t status;
  cublasHandle_t handle;
};



#endif

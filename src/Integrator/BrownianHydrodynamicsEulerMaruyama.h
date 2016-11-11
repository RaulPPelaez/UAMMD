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


#if defined SINGLE_PRECISION
#define cusolverDnpotrf cusolverDnSpotrf
#define cusolverDnpotrf_bufferSize cusolverDnSpotrf_bufferSize
#define cublastrmv cublasStrmv
#define cublassymv cublasSsymv
#define cublasnrm2 cublasSnrm2
#else
#define cusolverDnpotrf cusolverDnDpotrf
#define cusolverDnpotrf_bufferSize cusolverDnDpotrf_bufferSize
#define cublastrmv cublasDtrmv
#define curandGenerateNormal curandGenerateNormalDouble
#define cublassymv cublasDsymv
#define cublasnrm2 cublasDnrm2
#endif


namespace brownian_hy_euler_maruyama_ns{
  /*This virtual class takes D and computes a brownian noise array for each particle*/
  /*BrownianNoseComputer has at least a curand generator*/
  class BrownianNoiseComputer{
  public:
    BrownianNoiseComputer(uint N);  
    ~BrownianNoiseComputer();
    /*Initialize whatever you need according to D and N*/
    virtual bool init(real *D, uint N) = 0;
    /*Returns a pointer to the Brownian Noise vector, can be a pointer to noise i.e*/
    virtual real* compute(cublasHandle_t handle, real *D, uint N, cudaStream_t stream=0) = 0;
  protected:
    curandGenerator_t rng;
    Vector3 noise;
  };

  
  /*Computes the brownian noise using a cholesky decomposition on D, defined in cpp*/
  class BrownianNoiseCholesky: public BrownianNoiseComputer{
  public:
    BrownianNoiseCholesky(uint N): BrownianNoiseComputer(N){}
    bool init(real *D, uint N) override;
    real* compute(cublasHandle_t handle, real *D, uint N, cudaStream_t stream = 0) override;
  private:
    /*BdW is stored in the parents noise Vector3*/
    /*Cholesky decomposition through cuSolver*/
    cusolverDnHandle_t solver_handle;
    int h_work_size;
    real *d_work;
    int *d_info;
  };
  /*Computes the brownian noise using a Krylov subspace approximation from D \ref{1}, defined in cpp*/

  /*TODO TODO TODO TODO FINISH THIS DEFINE CONSTRUCTOR IN CPP*/
  class BrownianNoiseLanczos: public BrownianNoiseComputer{
  public:
    BrownianNoiseLanczos(uint N, uint max_iter=100);
    bool init(real *D, uint N) override;
    real* compute(cublasHandle_t handle, real *D, uint N, cudaStream_t stream = 0) override;
  private:
    /*BdW is stored in the parents noise Vector3*/    
    uint max_iter; //~100
    Vector3 w; //size N; v in each iteration
    Matrixf V; // 3Nxmax_iter; Krylov subspace base
    /*Matrix D in Krylov Subspace, stored as a vector because its dimension can change*/
    Vector<real> H; //size max_iter * max_iter;
    /*upper diagonal and diagonal of H, stored because size is unknown until Lanczos is complete*/
    Vector<real> hdiag, hsup; //size max_iter
    
    // void *cub_storage;
    // size_t cub_storage_size;
  };

}

enum StochasticNoiseMethod{CHOLESKY, LANCZOS};

class BrownianHydrodynamicsEulerMaruyama: public Integrator{
public:
  BrownianHydrodynamicsEulerMaruyama(Matrixf D0, Matrixf K,
				     StochasticNoiseMethod stochMethod = CHOLESKY);
				     
  ~BrownianHydrodynamicsEulerMaruyama();

  void update() override;
  real sumEnergy() override;
  
private:

  Vector3 force3; /*Cublas needs a real3 array instead of real4 to multiply matrices*/
  
  Vector3 DF;
  Matrixf D, K, D0;
  
  brownian_hy_euler_maruyama_ns::Params params;
  
  cudaStream_t stream, stream2;

  //cuSolverCholHandle cuChol;

  shared_ptr<brownian_hy_euler_maruyama_ns::BrownianNoiseComputer> cuBNoise;
  
  cublasStatus_t status;
  cublasHandle_t handle;
};



#endif

















// struct cuSolverCholHandle{
//   bool init(real *D, uint N){
//     cusolverDnCreate(&solver_handle);
//     h_work_size = 0;//work size of operation
    
//     cusolverDnpotrf_bufferSize(solver_handle, 
// 				CUBLAS_FILL_MODE_UPPER, 3*N, D, 3*N, &h_work_size);
//     gpuErrchk(cudaMalloc(&d_work, h_work_size*sizeof(real)));
//     gpuErrchk(cudaMalloc(&d_info, sizeof(int)));
//     this->D = D;
//     this->N = N;
    
//     return true;
//   }

//   bool compute(real *Dext=nullptr, uint Next=0, cudaStream_t stream = 0){
//     real *Dcomp = Dext;
//     uint Ncomp = Next;
//     if(!Dext) Dcomp = this->D;
//     if(!Next) Ncomp = this->N;

//     cusolverDnSetStream(solver_handle, stream);
//     cusolverDnpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
// 		      3*Ncomp, Dcomp, 3*Ncomp, d_work, h_work_size, d_info);
//     // int m_info;
//     // cudaMemcpy(&m_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
//     // cerr<<" "<<m_info<<endl;
//     return true;
//   }
  
//   real *D;
//   uint N;
//   cusolverDnHandle_t solver_handle;
//   int h_work_size;
//   real *d_work;
//   int *d_info;

  
  
// };

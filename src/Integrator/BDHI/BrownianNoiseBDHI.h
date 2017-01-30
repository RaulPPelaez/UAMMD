/*Raul P. Pealez 2016, part of the BrownianHydrodynamicsEulerMaruyama module.
  
  Computes an array of stochastic displacement depending on the diffusion matrix.

  AKA Computes BdW -> where B is such that B*B^T = M
  Where M is the mobility matrix and dW is a collection of independent standard Wiener processes.
*/
#ifndef BROWNIANNOISEBDHI_H
#define BROWNIANNOISEBDHI_H
#include"globals/defines.h"
#include"globals/globals.h"
#include "utils/utils.h"
#include<curand.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<cuda_runtime.h>
#include"utils/cuda_lib_defines.h"

#include"DiffusionBDHI.h"
namespace brownian_hy_euler_maruyama_ns{
  /*This virtual class takes D and computes a brownian noise array for each particle*/
  /*BrownianNoseComputer has at least a curand generator*/
  class BrownianNoiseComputer{
  public:
    BrownianNoiseComputer(uint N);/*Initialize curand*/
    ~BrownianNoiseComputer();
    /*Initialize whatever you need according to D and N*/
    virtual bool init(Diffusion &D, uint N) = 0;
    /*Returns a pointer to the Brownian Noise vector, can be a pointer to noise i.e*/
    virtual real* compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream=0) = 0;
    /*Fill a gpu array with normal noise*/
    real* genNoiseNormal(real mean, real std);
  protected:
    curandGenerator_t rng;
    Vector3 noise;
    real *noiseTemp;
    uint N;
  };

  /*-------------------------------Cholesky---------------------------------------*/
  /*Computes the brownian noise using a cholesky decomposition on D, defined in cpp*/
  class BrownianNoiseCholesky: public BrownianNoiseComputer{
  public:
    BrownianNoiseCholesky(uint N): BrownianNoiseComputer(N){}
    /*Initialize cuSolver*/
    bool init(Diffusion &D, uint N) override;
    /*Perform sqrt(D)·z by Cholesky decomposition and trmv multiplication*/
    real* compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream = 0) override;
  private:
    /*BdW is stored in the parents noise Vector3*/
    /*Cholesky decomposition through cuSolver*/
    cusolverDnHandle_t solver_handle;
    /*Cusolver temporal storage*/
    int h_work_size;    
    real *d_work;
    int *d_info;
  };
  /*--------------------------------Lanczos--------------------------------------*/
  /*Computes the brownian noise using a Krylov subspace approximation from D \ref{1}, defined in cpp*/
  class BrownianNoiseLanczos: public BrownianNoiseComputer{
  public:
    BrownianNoiseLanczos(uint N, uint max_iter=100);
    bool init(Diffusion &D, uint N) override;
    real* compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream = 0) override;
  protected:
    void compNoise(real z2, uint N, uint iter); //computes the noise in the current iteration
    /*BdW is stored in the parents noise Vector3*/    
    uint max_iter; //~100
    Vector3 w; //size N; v in each iteration
    Matrixf V; // 3Nxmax_iter; Krylov subspace base
    /*Matrix D in Krylov Subspace*/
    Matrixf H, Htemp; //size max_iter * max_iter;
    Matrixf P,Pt; //Transformation matrix to diagonalize H
    /*upper diagonal and diagonal of H*/
    Vector<real> hdiag, hdiag_temp, hsup; //size max_iter

    cusolverDnHandle_t solver_handle;
    cublasHandle_t cublas_handle;
    int h_work_size;
    real *d_work;
    int *d_info;   
  };
  

}
#endif

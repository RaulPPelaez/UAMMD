/*Raul P. Pelaez 2016. BDHI Cholesky submodule.
  
  Stores the full Mobility Matrix and computes the stochastic term via cholesky decomposition.

  sqrt(M)dW = B·dW -> D = B·B^T 

  It uses cuBLAS for Mv products and cuSOLVER for Cholesky decomposition
 */

#ifndef BDHI_CHOLESKY_CUH
#define BDHI_CHOLESKY_CUH

#include "BDHI.cuh"

namespace BDHI{
  class Cholesky: public BDHI_Method{
  public:
    Cholesky(real M0, real rh, int N);
    ~Cholesky();
    void setup_step(              cudaStream_t st = 0) override;
    void computeMF(real3* MF,     cudaStream_t st = 0) override;    
    void computeBdW(real3* BdW,   cudaStream_t st = 0) override;  
    void computeDivM(real3* divM, cudaStream_t st = 0) override;
    
    
  private:
    GPUVector<real> M; /*The full mobility matrix*/
    GPUVector3 force3;

    bool isMup2date;
    
    /*CUBLAS*/
    cublasStatus_t status;
    cublasHandle_t handle;
    /*CUSOLVER*/
    cusolverDnHandle_t solver_handle;
    /*Cusolver temporal storage*/
    int h_work_size;
    real *d_work;
    int *d_info;

    /*Kernel launch parameters*/
    int Nthreads, Nblocks;
    
    /*Rodne Prager Yamakawa device functions and parameters*/
    BDHI::RPYUtils utilsRPY;
  };
}
#endif

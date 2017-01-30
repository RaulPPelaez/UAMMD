/*Raul P. Pealez 2016, part of the BrownianHydrodynamicsEulerMaruyama module.
  Diffusion matrix handler

  Takes care of computing the mobility(diffusion) matrix,
    store it (if needed) and, most importantly, computing D·v

  The Diffusion matrix is computed according to the Rodne-Prager-Yamakawa tensor unless other is specified.
*/
#ifndef DIFFUSIONBDHI_H
#define DIFFUSIONBDHI_H
#include"globals/defines.h"
#include"globals/globals.h"
#include "utils/utils.h"
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<cuda_runtime.h>
#include"utils/cuda_lib_defines.h"
#include"DiffusionBDHIGPU.cuh"
#include<fstream>
namespace brownian_hy_euler_maruyama_ns{
  /*Virtual base class*/
  class Diffusion{
  protected:
    uint N;/*number of particles*/
    Matrixf D0;/*D0, self diffusion matrix, 3x3*/
    brownian_hy_euler_maruyama_ns::RPYParams params;
  public:
    Diffusion(Matrixf D0, uint N);    
    /*Fills the diffusion matrix, in a matrix-free method does nothing*/
    virtual void compute(){}

    /*res = D·v *//*D(3N,3N), v(3N), res(3N)*/
    virtual void dot(real *v, real *res, cublasHandle_t handle=0, cudaStream_t st = 0) = 0;

    /*Computes the divergence term when necesary, i.e. 2D*/
    void divergence(real *res, real *noise, cublasHandle_t handle=0, cudaStream_t st = 0);

    /*Returns nullptr in a Matrix-Free method*/
    virtual Matrixf* getMatrix(){return nullptr;}
  };

  /*Store a 3Nx3N matrix, compute D·v as a Matrix vector product*/
  class DiffusionFullMatrix: public Diffusion{
    Matrixf D;
  public:
    DiffusionFullMatrix(Matrixf D0, uint N);
    void compute() override;
    void dot(real *v, real *res, cublasHandle_t handle=0, cudaStream_t st = 0) override;
    Matrixf* getMatrix() override{return &(this->D);}
  };
  
  /*Do not store the diffusion matrix, recompute on the fly when computing D·v*/ 
  class DiffusionMatrixFree: public Diffusion{
  public:
    DiffusionMatrixFree(Matrixf D0, uint N);
    void dot(real *v, real *res, cublasHandle_t handle=0, cudaStream_t st = 0) override;

  };

}
#endif

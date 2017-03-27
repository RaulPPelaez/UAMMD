#ifndef LANCZOSALGORITHM_CUH
#define LANCZOSALGORITHM_CUH
#include<cuda.h>
#include<curand.h>
#include"utils/utils.h"
#include"globals/globals.h"
#include"utils/cuda_lib_defines.h"
#include"utils/utils.h"

/*To solveNoise, I need a functor which () operator is like this:

    inline void operator()(real3* Mv, real3 *v)
*/
struct LanczosAlgorithm{
  LanczosAlgorithm(){}
  LanczosAlgorithm(int N, int max_iter = 100);
  void init();
  ~LanczosAlgorithm(){
    //cublasDestroy(cublas_handle);    
  }
  template<class Dotctor>
  void solveNoise(Dotctor dot, real *BdW, cudaStream_t st = 0);

private:
  void compNoise(real z2, int N, int iter, real *BdW);
  int N;
  curandGenerator_t curng;
  /*CUBLAS*/
  cublasStatus_t status;
  cublasHandle_t cublas_handle;
  /*CUSOLVER*/
  cusolverDnHandle_t solver_handle;
  /*Cusolver temporal storage*/
  int h_work_size;
  real *d_work;
  int *d_info;
  /*Maximum number of Lanczos iterations*/
  int max_iter; //<100 in general

  /*Lanczos algorithm auxiliar memory*/
  GPUVector3 w; //size N, v in each iteration
  GPUVector<real> V; //size 3Nxmax_iter; Krylov subspace base transformation matrix
  //Mobility Matrix in the Krylov subspace
  Vector<real> H, Htemp; //Size max_iter x max_iter
  Vector<real> P, Pt;    //Transformation Matrix to diagonalize H, max_iter x max_iter
  /*upper diagonal and diagonal of H*/
  Vector<real> hdiag, hsup, hdiag_temp;
  Vector3 old_noise;
};


template<class Dotctor>
void LanczosAlgorithm::solveNoise(Dotctor dot, real *BdW, cudaStream_t st){

  cublasSetStream(cublas_handle, 0);
  cusolverDnSetStream(solver_handle, 0);
  curandSetStream(curng, 0);
  /*See J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347*/
      
  /************v[0] = z/||z||_2*****/
  /*Compute noise*/       /*V.d_m -> first column of V*/  
  curandGenerateNormal(curng, V.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  /*1/norm(z)*/
  real invz2; cublasnrm2(cublas_handle, 3*N, V.d_m, 1, &invz2); invz2 = 1.0/invz2;

  /*v[0] = v[0]*1/norm(z)*/ 
  cublasscal(cublas_handle, 3*N, &invz2,  V.d_m, 1);
  
  real alpha=1.0;
  /*Lanczos iterations for Krylov decomposition*/
  int i;
  for(i=0; i<max_iter; i++){
    /*w = D·vi ---> See NbodyFreeMatrixMobilityDot and Dotctor on how this works*/
    dot(w.d_m, (real3 *)(V.d_m+3*N*i)); 
    
    if(i>0){
      /*w = w-h[i-1][i]·vi*/
      alpha = -hsup[i-1];
      cublasaxpy(cublas_handle, 3*N,
		 &alpha,
		 V.d_m+3*N*(i-1), 1,
		 (real *)w.d_m, 1);
    }

    /*h[i][i] = dot(w, vi)*/
    cublasdot(cublas_handle, 3*N,
	      (real *)w.d_m, 1,
	      V.d_m+3*N*i, 1,
	      &(hdiag[i]));
    if(i<(int)max_iter-1){
      /*w = w-h[i][i]·vi*/
      alpha = -hdiag[i];
      cublasaxpy(cublas_handle, 3*N,
		 &alpha,
		 V.d_m+3*N*i, 1,
		 (real *)w.d_m, 1);
      /*h[i+1][i] = h[i][i+1] = norm(w)*/
      cublasnrm2(cublas_handle, 3*N, (real*)w.d_m, 1, &(hsup[i]));
      /*v_(i+1) = w·1/ norm(w)*/
      if(hsup[i]>real(0.0)){
	real invw2 = 1.0/hsup[i];
	cublasscal(cublas_handle, 3*N, &invw2,  (real *)w.d_m, 1);
      }
      else{/*If norm(w) = 0 that means all elements of w are zero, so set the first to 1*/
	real one = 1;
	cudaMemcpy(w.d_m, &one , sizeof(real), cudaMemcpyHostToDevice);
      }
      cudaMemcpy(V.d_m+3*N*(i+1), w.d_m, 3*N*sizeof(real), cudaMemcpyDeviceToDevice);
    }
  }
  
  /*Compute BdW using h and dw*/
  this->compNoise(1.0/invz2, N, i-1, (real *)BdW);
  
}

#endif
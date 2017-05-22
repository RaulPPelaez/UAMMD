/*Raul P. Pelaez 2017. Lanczos Algotihm,   
  Computes the matrix-vector product sqrt(M)·v recursively. In the case of solveNoise, v is a random gaussian vector.
  
  For that, it requires a functor in which the () operator takes an output real3* array and an input real3* (both device memory) as:
     inline __device__ operator()(real3* out, real3 * a_v);

  This function must fill out with the result of performing the M·v dot product- > out = M·a_v.

  If M has size NxN and the cost of the dot product is O(M). The total cost of the algorithm is O(m·M). Where m << N.

  If M·v performs a dense M-V product, the cost of the algorithm would be O(m·N^2).

References:

    [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
    J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347


TODO:
80- w can be V.d_m+ 3*N*(i+1), avoiding the cudaMemcpy at the end of each step
*/

#ifndef LANCZOSALGORITHM_CUH
#define LANCZOSALGORITHM_CUH
#include<cuda.h>
#include<curand.h>
#include"utils/utils.h"
#include"globals/globals.h"
#include"utils/cuda_lib_defines.h"
#include"utils/utils.h"
#include <nvToolsExt.h>
#include<fstream>
struct LanczosAlgorithm{
  LanczosAlgorithm(){}
  LanczosAlgorithm(int N, real tolerance = 1e-3);
  void init();
  ~LanczosAlgorithm(){
    cudaFree(d_work);
    cudaFree(d_info);
  }
  template<class Dotctor> //B = sqrt(M)
  void solve(Dotctor dot, real *Bv, real* v, cudaStream_t st = 0);

  template<class Dotctor>//B = sqrt(M)
  void solveNoise(Dotctor dot, real *BdW, cudaStream_t st = 0);

private:
  void compNoise(real z2, int N, int iter, real *BdW, cudaStream_t st = 0);
  void increment_max_iter();
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
  int min_iter; //Starting iteration guess
  real tolerance; //Desired error
  
  /*Lanczos algorithm auxiliar memory*/
  GPUVector3 w; //size N, v in each iteration
  GPUVector<real> V; //size 3Nxmax_iter; Krylov subspace base transformation matrix
  //Mobility Matrix in the Krylov subspace
  Vector<real> P;    //Transformation Matrix to diagonalize H, max_iter x max_iter
  /*upper diagonal and diagonal of H*/
  Vector<real> hdiag, hsup, htemp;
  GPUVector3 old_noise;

  int check_convergence_steps;
  
  bool initialized;
};


template<class Dotctor>
void LanczosAlgorithm::solve(Dotctor dot, real *Bz, real*z, cudaStream_t st){
  int steps_needed = 0;
  cublasSetStream(cublas_handle, st);
  cusolverDnSetStream(solver_handle, st);

  
  real normNoise_prev = 1.0; //For error estimation, see eq 27 in [1]

  /*See algorithm I in [1]*/
  /************v[0] = z/||z||_2*****/
  /*If v doesnt come from solveNoise*/
  if(z != V.d_m){
    cudaMemcpyAsync(V.d_m, z, 3*N*sizeof(real), cudaMemcpyDeviceToDevice, st);    
  }
  /*1/norm(z)*/
  real invz2; cublasnrm2(cublas_handle, 3*N, V.d_m, 1, &invz2); invz2 = 1.0/invz2;

  /*v[0] = v[0]*1/norm(z)*/ 
  cublasscal(cublas_handle, 3*N, &invz2,  V.d_m, 1);
  
  real alpha=1.0;
  /*Lanczos iterations for Krylov decomposition*/
  /*Will perform iterations until Error<=tolerance*/
  int i = -1;
  while(1){
    i++;
    /*w = D·vi ---> See i.e BDHI::Lanczos_ns::NbodyFreeMatrixMobilityDot and BDHI::Lanczos_ns::Dotctor on how this works*/
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

    /*Allocate more space if needed*/
    if(i==max_iter-1){
      cudaStreamSynchronize(st);
      this->increment_max_iter();
    }

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
	cudaMemcpyAsync(w.d_m, &one , sizeof(real), cudaMemcpyHostToDevice, st);
      }
      cudaMemcpyAsync(V.d_m+3*N*(i+1), w.d_m, 3*N*sizeof(real), cudaMemcpyDeviceToDevice, st);
    }

    /*Check convergence if needed*/
    steps_needed++;
    if(i >= check_convergence_steps && i>=3){ //Miminum of 3 iterations
#ifdef PROFILE_MODE
      nvtxRangePushA("COMP_NOISE");
#endif
      /*Compute Bz using h and z*/
      /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/      
      this->compNoise(1.0/invz2, N, i, (real *)Bz, st);  

      /*The first time the noise is computed it is only stored as old_noise*/
      if((i-check_convergence_steps)>0){
	/*Compute error as in eq 27 in [1]
	  Error = ||Bz_i - Bz_{i-1}||_2 / ||Bz_{i-1}||_2
	 */
	/*old_noise = Bz-old_noise*/
	real a=-1.0;	
	cublasaxpy(cublas_handle, 3*N,
		   &a,
		   Bz, 1,
		   (real*)old_noise.d_m, 1);

	/*yy = ||||Bz_i - Bz_{i-1}||_2*/
	real yy;	
	cublasnrm2(cublas_handle, 3*N, (real*) old_noise.d_m, 1, &yy);
	/*eq. 27 in [1]*/
	real Error = yy/normNoise_prev;
	//cerr<<Error<<endl;
	/*Convergence achieved!*/
	if(Error<=tolerance){
	  
	  //cerr<<"Tolerance reached!! "<<steps_needed<<endl;
	  // cerr<<"--------------------------------------"<<endl;
	  /*If I have needed more steps to converge than last time, 
	    start checking later for convergence next time*/
	  if(steps_needed-2 > check_convergence_steps){
	    check_convergence_steps += 1;
	    //cerr<<"Lanczos Convergence steps changed: "<<check_convergence_steps<<" "<<steps_needed<<endl;
	  }
	  /*Or check more often if I performed too many iterations*/
	  else{
	    check_convergence_steps -= 1;
	  }
#ifdef PROFILE_MODE
	  nvtxRangePop();
#endif
	  return;
	}
      }
      /*Always save the current noise as old_noise*/
      cudaMemcpyAsync(old_noise.d_m, Bz, N*sizeof(real3), cudaMemcpyDeviceToDevice, st);
      cublasnrm2(cublas_handle, 3*N, (real*) old_noise.d_m, 1, &normNoise_prev);

#ifdef PROFILE_MODE
      nvtxRangePop();
#endif      
    }


  }  

}

/*Computes sqrt(B)·dW, where dW is an array of gaussian random numbers*/
template<class Dotctor>
void LanczosAlgorithm::solveNoise(Dotctor dot, real *BdW, cudaStream_t st){
 
  /*Compute noise*/       /*V.d_m -> first column of V*/
  curandSetStream(curng, st);
  curandGenerateNormal(curng, V.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  this->solve(dot, BdW, V.d_m, st);

 
}

#endif


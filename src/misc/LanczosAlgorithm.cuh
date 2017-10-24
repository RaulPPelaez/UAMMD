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
#include"global/defines.h"
#include<cublas_v2.h>
#include"utils/cuda_lib_defines.h"
#include<thrust/device_vector.h>
#include"System/System.h"
#include"utils/debugTools.cuh"
#include"utils/cublasDebug.h"
#include<memory>
namespace uammd{
  enum class LanczosStatus{
    SUCCESS,   //Everything is fine
      SIZE_MISMATCH,  // V was asked with a certain size and provided to solve with a different one
      CUBLAS_ERROR,
      CUDA_ERROR //Error in memcpy, malloc ...
      };

  struct LanczosAlgorithm{  
    LanczosAlgorithm(shared_ptr<System> sys, real tolerance = 1e-3);
    void init();
    ~LanczosAlgorithm(){}

    //Fill the first N values of V and pass it to solve as "v" instead of an external array,
    //this will save a memcpy
    real * getV(int N){
      sys->log<System::DEBUG1>("[LanczosAlgorithm] V requested");
      if(N != this->N) numElementsChanged(N);
      return thrust::raw_pointer_cast(V.data());
    }

    //Given a Dotctor that computes a product M·v ( is handled by Dotctor ), computes Bv = sqrt(M)·v
    template<class Dotctor> //B = sqrt(M)
    LanczosStatus solve(Dotctor &dot, real *Bv, real* v, int N, real tolerance = 1e-3, cudaStream_t st = 0);

    LanczosStatus getLastError(){ return errorStatus; }
  private:
    void compResult(real z2, int N, int iter, real *BdW, cudaStream_t st = 0);
    //Increases storage space
    void increment_max_iter(int inc = 2);
    void numElementsChanged(int Nnew);
    int N;
    /*CUBLAS*/
    cublasHandle_t cublas_handle;
    /*Maximum number of Lanczos iterations*/
    int max_iter; //<100 in general, increases as needed
  
    /*Lanczos algorithm auxiliar memory*/
    thrust::device_vector<real3> w; //size N, v in each iteration
    thrust::device_vector<real> V; //size 3Nxmax_iter; Krylov subspace base transformation matrix
    //Mobility Matrix in the Krylov subspace
    thrust::host_vector<real> P;    //Transformation Matrix to diagonalize H, max_iter x max_iter
    /*upper diagonal and diagonal of H*/
    thrust::host_vector<real> hdiag, hsup, htemp;
    thrust::device_vector<real> htempGPU;
    thrust::device_vector<real3> oldBz;

    int check_convergence_steps;
  
    LanczosStatus errorStatus = LanczosStatus::SUCCESS;
    
    shared_ptr<System> sys;
  };

  
  
  template<class Dotctor>
  LanczosStatus LanczosAlgorithm::solve(Dotctor &dot, real *Bz, real*z, int N, real tolerance, cudaStream_t st){
    st = 0;
    sys->log<System::DEBUG1>("[LanczosAlgorithm] Computing sqrt(M)·v");
    //Exit if this instance has become boggus, in which case it should be reinitialized
    if(errorStatus != LanczosStatus::SUCCESS){
      return errorStatus;
    }

    //Handles the case of the number of elements changing since last call
    if(N != this->N){
      real * d_V = thrust::raw_pointer_cast(V.data());
      if(z == d_V){
	errorStatus = LanczosStatus::SIZE_MISMATCH;
	return errorStatus;
      }      
      numElementsChanged(N);
    }

    real * d_V = thrust::raw_pointer_cast(V.data());
    int steps_needed = 0;
    CublasSafeCall(cublasSetStream(cublas_handle, st));
  
    sys->log<System::DEBUG2>("[LanczosAlgorithm] Starting");
    real normNoise_prev = 1.0; //For error estimation, see eq 27 in [1]

    /*See algorithm I in [1]*/
    /************v[0] = z/||z||_2*****/
  
    /*If v doesnt come from solveNoise*/
    if(z != d_V){
      sys->log<System::DEBUG2>("[LanczosAlgorithm] Copying input to subspace  proyection matrix");
      CudaSafeCall(cudaMemcpyAsync(d_V, z, 3*N*sizeof(real), cudaMemcpyDeviceToDevice, st));
    }
    /*1/norm(z)*/
    real invz2;
    CublasSafeCall(cublasnrm2(cublas_handle, 3*N, d_V, 1, &invz2));
    invz2 = 1.0/invz2;

    /*v[0] = v[0]*1/norm(z)*/ 
    CublasSafeCall(cublasscal(cublas_handle, 3*N, &invz2,  d_V, 1));
  
    real alpha=1.0;
    /*Lanczos iterations for Krylov decomposition*/
    /*Will perform iterations until Error<=tolerance*/
    int i = -1;
    while(errorStatus == LanczosStatus::SUCCESS){
      i++;
      real3 * d_w = thrust::raw_pointer_cast(w.data());
      sys->log<System::DEBUG3>("[LanczosAlgorithm] Iteration %d", i);
      /*w = D·vi ---> See i.e BDHI::Lanczos_ns::NbodyFreeMatrixMobilityDot and BDHI::Lanczos_ns::Dotctor on how this works*/
      sys->log<System::DEBUG3>("[LanczosAlgorithm] Computing M·v");
      dot(d_w, (real3 *)(d_V+3*N*i));
      
      if(i>0){
	/*w = w-h[i-1][i]·vi*/
	alpha = -hsup[i-1];
	CublasSafeCall(cublasaxpy(cublas_handle, 3*N,
				  &alpha,
				  d_V+3*N*(i-1), 1,
				  (real *)d_w, 1));
      }

      /*h[i][i] = dot(w, vi)*/
      CublasSafeCall(cublasdot(cublas_handle, 3*N,
		(real *)d_w, 1,
		d_V+3*N*i, 1,
		&(hdiag[i])));
      sys->log<System::DEBUG4>("[LanczosAlgorithm] hdiag[%d] %f", i, hdiag[i]);
      /*Allocate more space if needed*/
      if(i == max_iter-1){
	CudaSafeCall(cudaStreamSynchronize(st));
	this->increment_max_iter();
	d_V = thrust::raw_pointer_cast(V.data());
	d_w = thrust::raw_pointer_cast(w.data());
      }

      if(i<(int)max_iter-1){
	/*w = w-h[i][i]·vi*/
	alpha = -hdiag[i];
	CublasSafeCall(cublasaxpy(cublas_handle, 3*N,
				  &alpha,
				  d_V+3*N*i, 1,
				  (real *)d_w, 1));
	/*h[i+1][i] = h[i][i+1] = norm(w)*/
	CublasSafeCall(cublasnrm2(cublas_handle, 3*N, (real*)d_w, 1, &(hsup[i])));
	/*v_(i+1) = w·1/ norm(w)*/
	if(hsup[i]>real(0.0)){
	  real invw2 = 1.0/hsup[i];
	  CublasSafeCall(cublasscal(cublas_handle, 3*N, &invw2,  (real *)d_w, 1));
	}
	else{/*If norm(w) = 0 that means all elements of w are zero, so set the first to 1*/
	  real one = 1;
	  CudaSafeCall(cudaMemcpyAsync(d_w, &one , sizeof(real), cudaMemcpyHostToDevice, st));
	}
	CudaSafeCall(cudaMemcpyAsync(d_V+3*N*(i+1), d_w, 3*N*sizeof(real), cudaMemcpyDeviceToDevice, st));
      }
      
      /*Check convergence if needed*/
      steps_needed++;
      if(i >= check_convergence_steps && i>=3){ //Miminum of 3 iterations
	sys->log<System::DEBUG3>("[LanczosAlgorithm] Checking convergence");
	/*Compute Bz using h and z*/
	/**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/
	this->compResult(1.0/invz2, N, i, (real *)Bz, st);  

	/*The first time the result is computed it is only stored as oldBz*/
	if((i-check_convergence_steps)>0){
	  /*Compute error as in eq 27 in [1]
	    Error = ||Bz_i - Bz_{i-1}||_2 / ||Bz_{i-1}||_2
	  */
	  /*oldBz = Bz-oldBz*/
	  real * d_oldBz = (real*) thrust::raw_pointer_cast(oldBz.data());
	  real a=-1.0;	
	  CublasSafeCall(cublasaxpy(cublas_handle, 3*N,
				    &a,
				    Bz, 1,
				    d_oldBz, 1));

	  /*yy = ||||Bz_i - Bz_{i-1}||_2*/
	  real yy;	  
	  CublasSafeCall(cublasnrm2(cublas_handle, 3*N,  d_oldBz, 1, &yy));
	  //eq. 27 in [1]
	  real Error = yy/normNoise_prev;	  
	  //Convergence achieved!
	  if(Error <= tolerance){
	    if(steps_needed-2 > check_convergence_steps){
	      check_convergence_steps += 1;	      
	    }
	    //Or check more often if I performed too many iterations
	    else{
	      check_convergence_steps -= 1;
	    }
	    sys->log<System::DEBUG1>("[LanczosAlgorithm] Convergence in %d iterations with error %f",i, Error);
	    return errorStatus;
	  }
	  else{
	    sys->log<System::DEBUG3>("[LanczosAlgorithm] Convergence not achieved! Error: %f, Tolerance: %f", Error, tolerance);
	    sys->log<System::DEBUG3>("[LanczosAlgorithm] yy: %f, normNoise_prev: %f", yy, normNoise_prev);
	  }
	}
	sys->log<System::DEBUG3>("[LanczosAlgorithm] Saving current result.");
	/*Always save the current result as oldBz*/
	real * d_oldBz = (real*) thrust::raw_pointer_cast(oldBz.data());
	CudaSafeCall(cudaMemcpyAsync(d_oldBz, Bz, N*sizeof(real3), cudaMemcpyDeviceToDevice, st));
	CublasSafeCall(cublasnrm2(cublas_handle, 3*N, (real*) d_oldBz, 1, &normNoise_prev));

#ifdef USE_NVTX
	nvtxRangePop();
#endif      
      }
    }
    return errorStatus;
  }

}
#include"LanczosAlgorithm.cu"
#endif


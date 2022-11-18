/*Raul P. Pelaez 2017-2022. Lanczos algorithm

  References:
  [1] Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations.
  -http://dx.doi.org/10.1063/1.4742347

*/
#include"global/defines.h"
#include"misc/LanczosAlgorithm.cuh"
#include <limits>
#include<string.h>
#include "device_container.h"
#include"misc/lapack_and_blas_defines.h"
#include"device_blas.h"
#include<cmath>
#include<stdexcept>
#include"utils/debugTools.h"

#include<limits>
#include <string>
namespace uammd{
  namespace lanczos{

    namespace detail{

      /*See algorithm I in [1]*/
      class KrylovSubspace{
	cublasHandle_t cublas_handle;
	cudaStream_t st = 0;
	device_container<real> w; //size N, v in each iteration
	device_container<real> V; //size Nxmax_iter; Krylov subspace base transformation matrix
	//Mobility Matrix in the Krylov subspace
	std::vector<real> P;    //Transformation Matrix to diagonalize H, max_iter x max_iter
	/*upper diagonal and diagonal of H*/
	std::vector<real> hdiag, hsup, htemp;
	device_container<real> htempGPU;
	int N;
	int subSpaceDimension;

	real normz;
	void diagonalizeSubSpace(){
	  int size = getSubSpaceSize();
	  /**************LAPACKE********************/
	  /*The tridiagonal matrix is stored only with its diagonal and subdiagonal*/
	  /*Store both in a temporal array*/
	  for(int i=0; i<size; i++){
	    htemp[i] = hdiag[i];
	    htemp[i+size]= hsup[i];
	  }
	  /*P = eigenvectors must be filled with zeros, I do not know why*/
	  real* h_P = P.data();
	  memset(h_P, 0, size*size*sizeof(real));
	  /*Compute eigenvalues and eigenvectors of a triangular symmetric matrix*/
	  auto info = LAPACKE_steqr(LAPACK_COL_MAJOR, 'I',
				    size, &htemp[0], &htemp[0]+size,
				    h_P, size);
	  if(info!=0){
	    throw std::runtime_error("[Lanczos] Could not diagonalize tridiagonal krylov matrix, steqr failed with code " + std::to_string(info));
	  }
	}

	real *computeSquareRoot(){
	  int size = getSubSpaceSize();
	  diagonalizeSubSpace();
	  /***Hdiag_temp = Hdiag·P·e1****/
	  for(int j=0; j<size; j++){
	    htemp[j] = sqrt(htemp[j])*P[size*j];
	  }
	  /***** Htemp = H^1/2·e1 = Pt· hdiag_temp ****/
	  /*Compute with blas*/
	  real* h_P = P.data();
	  real alpha = 1.0;
	  real beta = 0.0;
	  cblas_gemv(CblasColMajor, CblasNoTrans,
		     size, size,
		     alpha,
		     h_P, size,
		     &htemp[0], 1,
		     beta,
		     &htemp[0]+size, 1);
	  detail::device_copy(htemp.begin()+size, htemp.begin()+2*size, htempGPU.begin());
	  return detail::getRawPointer(htempGPU);
	}

	real *getTransformationMatrix(){
	  return detail::getRawPointer(V);
	}

	void resize(int subSpaceSize){
	  CudaSafeCall(cudaDeviceSynchronize());
	  try{
	    w.resize((N+1), real());
	    V.resize(N*subSpaceSize,0);
	    P.resize(subSpaceSize*subSpaceSize,0);
	    hdiag.resize(subSpaceSize+1,0);
	    hsup.resize(subSpaceSize+1,0);
	    htemp.resize(2*subSpaceSize,0);
	    htempGPU.resize(2*subSpaceSize,0);
	  }
	  catch(...){
	    throw std::runtime_error("[KrylovSubspace] Could not allocate memory");
	  }
	}

      public:
	KrylovSubspace(int N):subSpaceDimension(0), N(N){
	  //Init cuBLAS for Lanczos process
	  CublasSafeCall(cublasCreate_v2(&cublas_handle));
	  this->resize(1);
	}

	~KrylovSubspace(){
	  CublasSafeCall(cublasDestroy(cublas_handle));
	}

	/************v[0] = z/||z||_2*****/
	void setFirstBasisVector(const real* z){
	  /*1/norm(z)*/
	  real* Vm = getTransformationMatrix();
	  lanczos::device_nrm2(cublas_handle, N, z, 1, &(this->normz));
	  //this->normz = computeNorm(z, N);
	  /*v[0] = v[0]*1/norm(z)*/
	  real invz2 = 1.0/normz;
	  thrust::device_ptr<const real> zp(z);
	  detail::device_copy(zp, zp+N, thrust::device_ptr<real>(Vm));
	  lanczos::device_scal(cublas_handle, N, &invz2, Vm, 1);
	}

	void nextIteration(MatrixDot* dot){
	  int i = subSpaceDimension;
	  this->subSpaceDimension++;
	  resize(subSpaceDimension+1);
	  auto d_V = detail::getRawPointer(V);
	  auto d_w = detail::getRawPointer(w);
	  /*w = D·vi*/
	  dot->setSize(N);
	  dot->operator()(d_V+N*i, d_w);
	  if(i>0){
	    /*w = w-h[i-1][i]·vi*/
	    real alpha = -hsup[i-1];
	    device_axpy(cublas_handle, N,
			&alpha,
			d_V+N*(i-1), 1,
			d_w, 1);
	  }
	  /*h[i][i] = dot(w, vi)*/
	  device_dot(cublas_handle, N,
		     d_w, 1,
		     d_V+N*i, 1,
		     &(hdiag[i]));
	  /*w = w-h[i][i]·vi*/
	  real alpha = -hdiag[i];
	  device_axpy(cublas_handle, N,
		      &alpha,
		      d_V+N*i, 1,
		      d_w, 1);
	  /*h[i+1][i] = h[i][i+1] = norm(w)*/
	  device_nrm2(cublas_handle, N, (real*)d_w, 1, &(hsup[i]));
	  /*v_(i+1) = w·1/ norm(w)*/
	  real tol = 1e-3*hdiag[i]/normz;
	  if(hsup[i]<tol) hsup[i] = real(0.0);
	  if(hsup[i]>real(0.0)){
	    real invw2 = 1.0/hsup[i];
	    device_scal(cublas_handle, N, &invw2, d_w, 1);
	  }
	  else{/*If norm(w) = 0 that means all elements of w are zero, so set w = e1*/
	    detail::device_fill(w.begin(), w.end(), real());
	    w[0] = 1;
	  }
	  detail::device_copy(w.begin(), w.begin()+N, V.begin() + N*(i+1));
	}

	int getSubSpaceSize(){
	  return subSpaceDimension;
	}

	//Computes the current result guess sqrt(M)·v, stores in BdW
	void computeCurrentResultEstimation(real *BdW){
	  int m = getSubSpaceSize();
	  /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/
	  /**** H^1/2·e1 = Pt· first_column_of(sqrt(Hdiag)·P) ******/
	  real* HhalfDotE1 = computeSquareRoot();
	  /*y = ||z||_2 * Vm · H^1/2 · e1 = Vm · (z2·hdiag_temp)*/
	  real *Vm = getTransformationMatrix();
	  real beta = 0.0;
	  device_gemv(cublas_handle, N, m,
		      &this->normz,
		      Vm, N,
		      HhalfDotE1, 1,
		      &beta,
		      BdW, 1);
	}
      };
    }

    Solver::Solver():
      check_convergence_steps(3){
      //Init cuBLAS for Lanczos process
      CublasSafeCall(cublasCreate_v2(&cublas_handle));
    }

    Solver::~Solver(){
      CublasSafeCall(cublasDestroy(cublas_handle));
    }

    real Solver::runIterations(MatrixDot *dot, real *Bz, const real*z, int numberIterations, int N){
      oldBz.resize((N+1), real());
      /*Lanczos iterations for Krylov decomposition*/
      detail::KrylovSubspace solver(N);
      solver.setFirstBasisVector(z);
      for(int i = 0; i<numberIterations; i++){
	solver.nextIteration(dot);
	if(i==numberIterations-2){
	  solver.computeCurrentResultEstimation(Bz);
	  thrust::device_ptr<real> Bzp(Bz);
	  detail::device_copy(Bzp, Bzp+N, oldBz.begin());
	}
      }
      solver.computeCurrentResultEstimation(Bz);
      real currentResidual =  computeError(Bz, N, numberIterations);
      return currentResidual;
    }

    int Solver::run(MatrixDot *dot, real *Bz, const real*z, real tolerance, int N, cudaStream_t st){
      oldBz.resize((N+1));
      detail::device_fill(oldBz.begin(), oldBz.end(), real());
      /*Lanczos iterations for Krylov decomposition*/
      detail::KrylovSubspace solver(N);
      solver.setFirstBasisVector(z);
      const int checkConvergenceSteps = std::min(check_convergence_steps, iterationHardLimit-2);
      for(int i = 0; i<iterationHardLimit; i++){
	solver.nextIteration(dot);
	if(i>=checkConvergenceSteps){
	  solver.computeCurrentResultEstimation(Bz);
	  if(i>0){
	    auto currentResidual = computeError(Bz, N, i);
	    if(currentResidual<=tolerance){
	      registerRequiredStepsForConverge(i);
	      return i;
	    }
	  }
	  //Store current estimation
	  thrust::device_ptr<real> Bzp(Bz);
	  detail::device_copy(Bzp, Bzp+N, oldBz.begin());
	}
      }
      throw std::runtime_error("[Lanczos] Could not converge");
    }

    real Solver::computeError(real *Bz, int N, int iter){
      /*Compute error as in eq 27 in [1]
	Error = ||Bz_i - Bz_{i-1}||_2 / ||Bz_{i-1}||_2
      */
      real normResult_prev;
      real * d_oldBz = detail::getRawPointer(oldBz);
      device_nrm2(cublas_handle, N, d_oldBz, 1, &normResult_prev);
      /*oldBz = Bz-oldBz*/
      real a=-1.0;
      device_axpy(cublas_handle, N,
		  &a,
		  Bz, 1,
		  d_oldBz, 1);
      /*yy = ||Bz_i - Bz_{i-1}||_2*/
      real yy;
      device_nrm2(cublas_handle, N,  d_oldBz, 1, &yy);
      //eq. 27 in [1]
      real Error = abs(yy/normResult_prev);
      if(std::isnan(Error)){
	throw std::runtime_error("[Lanczos] Unknown error (found NaN in result guess) at iteration "
				 + std::to_string(iter));
      }
      return Error;
    }

    void Solver::registerRequiredStepsForConverge(int steps_needed){
      this->lastRunRequiredSteps = steps_needed;
      if(steps_needed-2 > check_convergence_steps){
	check_convergence_steps += 1;
      }
      //Or check more often if I performed too many iterations
      else{
	check_convergence_steps = std::max(1, check_convergence_steps - 2);
      }
    }

  }
}

/*Raul P. Pelaez. 2017. Cholesky BDHI submodule implementation

  Computes the hydrodynamic interactions between particles in the system by
  maintaining a 3Nx3N mobility matrix with the RPY tensor in memory
  and explicitly computing M·F as a matrix-vector product.

  Note that only the upper part of M is stored, as M is symmetric.

  The brownian noise is computed as BdW = chol(M)·dW with cuSOLVER and cuBLAS

References:
[1] https://github.com/RaulPPelaez/UAMMD/wiki/BDHI_Cholesky
[2] https://github.com/RaulPPelaez/UAMMD/wiki/NBody-Forces
*/
#include"BDHI_Cholesky.cuh"
#include"misc/Transform.cuh"
#include"Interactor/NBodyForces.cuh"
#include<fstream>
using namespace std;
using namespace BDHI;

namespace Cholesky_ns{
/*Fills the 3Nx3N Mobility matrix with Mij = RPY(|rij|)(I-r^r) in blocks of 3x3 matrices*/
  __global__ void fillMobilityRPYD(real * __restrict__ M,
				   const  real4* __restrict__ R,
				   uint N,
				   real M0, BDHI::RPYUtils utils){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;
    uint n = 3*N;

    /*Self Diffusion*/
    
    for(int k = 0; k < 3; k++)
      for(int l = 0; l < 3; l++){
	M[3*i + k + n*(3*i + l)] =  real(0.0);
      }
    M[3*i + 0 + n*(3*i + 0)] = M0;
    M[3*i + 1 + n*(3*i + 1)] = M0;
    M[3*i + 2 + n*(3*i + 2)] = M0;
  
    real3 rij;
    real* rijp = &(rij.x);    
    real c1, c2;
    for(int j=i+1; j<N; j++){	
      rij = make_real3(R[j]) - make_real3(R[i]);
#ifdef SINGLE_PRECISION
      const real r = sqrtf(dot(rij, rij));
#else
      const real r = sqrt(dot(rij, rij));
#endif
      const real invr2 = 1.0/(r*r);
      /*Rotne-Prager-Yamakawa tensor */
      const real2 c12 = utils.RPY(r);      
      c1 = M0*c12.x;
      c2 = M0*c12.y*invr2;
      /*3x3 Matrix for each particle pair*/
      for(int k = 0; k < 3; k++)
 	for(int l = 0; l < 3; l++)
	  M[3*i + k + n*(3*j + l)] = c2*rijp[k]*rijp[l];
      /*Diagonal*/
      for(int k = 0; k<3; k++)  M[3*i + k + n*(3*j + k)] += c1;
    }
  }
}

Cholesky::Cholesky(real M0, real rh, int N):
  BDHI_Method(M0, rh, N), utilsRPY(rh), force3(N), M(3*N*3*N){
  cerr<<"\tInitializing Cholesky subsystem...";  
 
  M.memset(0);

  BLOCKSIZE = 128;
  Nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  Nblocks  =  N/Nthreads +  ((N%Nthreads!=0)?1:0); 

  /*Init cuSolver for BdW*/
  cusolverDnCreate(&solver_handle);
  h_work_size = 0;//work size of operation

  cusolverDnpotrf_bufferSize(solver_handle, 
			     CUBLAS_FILL_MODE_UPPER, 3*N, M.d_m, 3*N, &h_work_size);
  gpuErrchk(cudaMalloc(&d_work, h_work_size));
  gpuErrchk(cudaMalloc(&d_info, sizeof(int)));
  /*Init cuBLAS for MF*/ 
  status = cublasCreate(&handle);
  if(status){cerr<<"ERROR with CUBLAS!!\n"<<endl; exit(1);}

  isMup2date = false;
  cerr<<"DONE!!"<<endl;  
}

  
Cholesky::~Cholesky(){
    cublasDestroy(handle);
    cudaFree(d_work);
    cudaFree(d_info);
}


void Cholesky::setup_step(cudaStream_t st){
  /*Fill the upper part of symmetric mobility matrix*/
  Cholesky_ns::fillMobilityRPYD<<<Nblocks, Nthreads, 0 ,st>>>(M.d_m, pos.d_m, N,
							      M0, utilsRPY);
  /*M contains the mobility tensor in this step*/
  isMup2date = true;
}

/*This little functor is used with Transform::transform and changes a real4 to a real3*/
// See https://github.com/RaulPPelaez/UAMMD/wiki/Utils
namespace Cholesky_ns{
  struct changetor{
    inline __device__ real3 operator()(real4 t) const{ return make_real3(t);}
  };
}


void Cholesky::computeMF(real3* MF, cudaStream_t st){
  /*computeMF should be called before computeBdW*/
  static bool warning_printed = false;
  if(!isMup2date){
    setup_step(st);
    if(!warning_printed){
      cerr<<"WARNING!!: You should call computeMF inmediatly after setup_step, otherwise M will be compute twice per step!"<<endl;
      warning_printed = true;
    }
  }

  /*Morphs a real4 vector into a real3 one, needed by cublas*/
  cublasSetStream(handle, st);
  Transform::transform<<<Nblocks, Nthreads, 0, st>>>(force.d_m,
						     force3.d_m,
						     Cholesky_ns::changetor(), N);
  
  real alpha = 1.0;
  real beta = 0;
  /*Compute M·F*/
  cublassymv(handle, CUBLAS_FILL_MODE_UPPER,
	     3*N, 
	     &alpha,
	     M.d_m, 3*N,
	     (real*) force3.d_m, 1,
	     &beta,
	     (real*)MF, 1); 
}


void Cholesky::computeBdW(real3 *BdW, cudaStream_t st){
  if(!isMup2date) setup_step();
  /*computeBdw messes up M, fills it with B*/
  isMup2date = false;
  cusolverDnSetStream(solver_handle, st);
  
  /*Perform cholesky factorization, store B on LOWER part of M matrix*/
  cusolverDnpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		  3*N, M.d_m, 3*N, d_work, h_work_size, d_info);
  curandSetStream(curng, st);
  /*Gen new noise in BdW*/
  curandGenerateNormal(curng, (real*) BdW, 3*N + ((3*N)%2), real(0.0), real(1.0));

  cublasSetStream(handle, st);
  /*Compute B·dW -> y = M·y*/
  cublastrmv(handle, //B is an upper triangular matrix (with non unit diagonal)
	     CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
	     3*N,
	     M.d_m, 3*N,
	     (real*)BdW, 1);

}


namespace Cholesky_ns{
  /*Exactly the same as Lanczos_ns::divMTranverser.
    It is placed here for convinience when performing tests that involve 
    changing the input parameters to the class*/
  /*This Nbody Transverser computes the analytic divergence of the RPY tensor*/
  // https://github.com/RaulPPelaez/UAMMD/wiki/Nbody-Forces
  // https://github.com/RaulPPelaez/UAMMD/wiki/Transverser
  struct divMTransverser{
    divMTransverser(real3* divM, real M0, real rh): divM(divM), M0(M0), rh(rh){
      this->invrh = 1.0/rh;
    }
    
    inline __device__ real3 zero(){ return make_real3(real(0.0));}
    inline __device__ real3 compute(const real4 &pi, const real4 &pj){
      /*Work in units of rh*/
      const real3 r12 = (make_real3(pi)-make_real3(pj))*invrh;
      const real r2 = dot(r12, r12);
      if(r2==real(0.0))
	return make_real3(real(0.0));
#ifdef SINGLE_PRECISION
      real invr = rsqrtf(r2);
#else
      real invr = rsqrt(r2);
#endif
      /*Just the divergence of the RPY tensor in 2D, taken from A. Donev's notes*/
      /*The 1/6pia is in M0, the factor kT is in the integrator, and the factor 1/a is in set*/
      if(r2>real(4.0)){
	real invr2 = invr*invr;
	return real(0.75)*(r2-real(2.0))*invr2*invr2*r12*invr;
      }
      else{
	return real(0.09375)*r12*invr;
      }
    }
    inline __device__ void accumulate(real3 &total, const real3 &cur){total += cur;}
    
    inline __device__ void set(int id, const real3 &total){
      divM[id] = M0*total*invrh;
    }
  private:
    real3* divM;
    real M0;
    real rh, invrh;
  };

}

void Cholesky::computeDivM(real3* divM, cudaStream_t st){
  /*A simple NBody transverser, see https://github.com/RaulPPelaez/UAMMD/wiki/NBody-Forces */
   Cholesky_ns::divMTransverser divMtr(divM, M0, utilsRPY.rh);
  
   NBodyForces<Cholesky_ns::divMTransverser> nbody_divM(divMtr, st);
  
   nbody_divM.sumForce();
}
 
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
#include"Interactor/NBody.cuh"
#include"utils/GPUUtils.cuh"
#include"utils/debugTools.cuh"
#include<cublas_v2.h>
#include"utils/cublasDebug.h"
#include<cusolverDn.h>
#include"utils/cusolverDebug.h"

#include<utils/cuda_lib_defines.h>

#include<fstream>
namespace uammd{
  namespace BDHI{

    namespace Cholesky_ns{
      template<class IndexIter>
      /*Fills the 3Nx3N Mobility matrix with Mij = RPY(|rij|)(I-r^r) in blocks of 3x3 matrices*/
      __global__ void fillMobilityRPYD(real * __restrict__ M,
				       const  real4* __restrict__ R,
				       IndexIter indexIter,
				       uint N,
				       real M0, BDHI::RotnePragerYamakawa rpy){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=N) return;
	int i = indexIter[id];
	uint n = 3*N;

	/*Self Diffusion*/
    
	for(int k = 0; k < 3; k++)
	  for(int l = 0; l < 3; l++){
	    M[3*id + k + n*(3*id + l)] =  real(0.0);
	  }
	M[3*id + 0 + n*(3*id + 0)] = M0;
	M[3*id + 1 + n*(3*id + 1)] = M0;
	M[3*id + 2 + n*(3*id + 2)] = M0;
  
	real3 rij;
	real* rijp = &(rij.x);    
	real c1, c2;
	real3 ri = make_real3(R[i]);
	for(int j=id+1; j<N; j++){

	  rij = make_real3(R[indexIter[j]]) - ri;
#ifdef SINGLE_PRECISION
	  const real r = sqrtf(dot(rij, rij));
#else
	  const real r = sqrt(dot(rij, rij));
#endif
	  const real invr2 = real(1.0)/(r*r);
	  /*Rotne-Prager-Yamakawa tensor */
	  const real2 c12 = rpy(r);      
	  c1 = M0*c12.x;
	  c2 = M0*c12.y*invr2;
	  /*3x3 Matrix for each particle pair*/
	  for(int k = 0; k < 3; k++)
	    for(int l = 0; l < 3; l++)
	      M[3*id + k + n*(3*j + l)] = c2*rijp[k]*rijp[l];
	  /*Diagonal*/
	  for(int k = 0; k<3; k++)  M[3*id + k + n*(3*j + k)] += c1;
	}
      }
    }

    Cholesky::Cholesky(shared_ptr<ParticleData> pd,
		       shared_ptr<ParticleGroup> pg,
		       shared_ptr<System> sys,		       
		       Parameters par):
      pd(pd), pg(pg), sys(sys),
      par(par),
      rpy(par.hydrodynamicRadius){

      sys->log<System::MESSAGE>("[BDHI::Cholesky] Initialized");  
 
      int numberParticles = pg->getNumberParticles();

      force3.resize(numberParticles, real3());
      mobilityMatrix.resize(pow(3*numberParticles,2)+1, real());

      this->selfMobility = 1.0/(6*M_PI*par.viscosity*par.hydrodynamicRadius);
      
      /*Init cuSolver for BdW*/
      CusolverSafeCall(cusolverDnCreate(&solver_handle));
      h_work_size = 0;//work size of operation

      auto d_M = thrust::raw_pointer_cast(mobilityMatrix.data());
      CusolverSafeCall(cusolverDnpotrf_bufferSize(solver_handle, 
				 CUBLAS_FILL_MODE_UPPER,
				 3*numberParticles,
				 d_M, 3*numberParticles,
						  &h_work_size));
      CudaSafeCall(cudaMalloc(&d_work, h_work_size));
      CudaSafeCall(cudaMalloc(&d_info, sizeof(int)));
      /*Init cuBLAS for MF*/ 
      CublasSafeCall(cublasCreate(&handle));


      
      /*Create noise*/
      curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next());
      /*Create a temporal vector to warm up curand*/
      thrust::device_vector<real> noise(numberParticles+1);
      //Curand fill with gaussian numbers with mean 0 and var 1
      /*This shit is obscure, curand will only work with an even number of elements*/
      auto d_noise = thrust::raw_pointer_cast(noise.data());
      curandGenerateNormal(curng, d_noise, 3*numberParticles + ((3*numberParticles)%2), real(0.0), real(1.0));   
      
      isMup2date = false;      
    }

  
    Cholesky::~Cholesky(){
      cublasDestroy(handle);
      curandDestroyGenerator(curng);
      cudaFree(d_work);
      cudaFree(d_info);
    }


    void Cholesky::init(){}
    
    void Cholesky::setup_step(cudaStream_t st){

      sys->log<System::DEBUG3>("[BDHI::Cholesky] Setup Step");  
      int numberParticles = pg->getNumberParticles();

      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto groupIter = pg->getIndexIterator(access::location::gpu);

      auto d_M = thrust::raw_pointer_cast(mobilityMatrix.data());

      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
      /*Fill the upper part of symmetric mobility matrix*/
      Cholesky_ns::fillMobilityRPYD<<<Nblocks, Nthreads, 0 ,st>>>(d_M,
       								  pos.raw(),
       								  groupIter,
       								  numberParticles,
       								  selfMobility, rpy);
      /*M contains the mobility tensor in this step*/
      isMup2date = true;
    }

    namespace Cholesky_ns{
      template<class IndexIter, class Real3OutputIterator>
      __global__ void real4ToReal3(IndexIter indexIter,
				   real4 * in,
				   Real3OutputIterator out,
				   int N){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=N) return;
	int i = indexIter[id];
	out[id] = make_real3(in[i]);	
      }
				   
    }

    void Cholesky::computeMF(real3* MF, cudaStream_t st){
      sys->log<System::DEBUG3>("[BDHI::Cholesky] MF");  
      /*computeMF should be called before computeBdW*/
      static bool warning_printed = false;
      if(!isMup2date){
	setup_step(st);
	if(!warning_printed){
	  sys->log<System::WARNING>("[BDHI::Cholesky] You should call computeMF inmediatly after setup_step, otherwise M will be compute twice per step!");
	  warning_printed = true;
	}
      }

      int numberParticles = pg->getNumberParticles();
      /*Morphs a real4 vector into a real3 one, needed by cublas*/
      cublasSetStream(handle, st);
      
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      auto indexIter = pg->getIndexIterator(access::location::gpu);
      
            
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);

      Cholesky_ns::real4ToReal3<<<Nblocks, Nthreads, 0 , st>>>(indexIter,
							       force.raw(),
							       force3.begin(),
							       numberParticles);
						  
      real alpha = 1.0;
      real beta = 0;
      /*Compute M·F*/
      real * d_M = thrust::raw_pointer_cast(mobilityMatrix.data());
      real * d_force3 = (real*)thrust::raw_pointer_cast(force3.data());
      
      cublassymv(handle, CUBLAS_FILL_MODE_UPPER,
		 3*numberParticles, 
		 &alpha,
		 d_M, 3*numberParticles,
		 d_force3, 1,
		 &beta,
		 (real *)MF, 1); 
    }


    void Cholesky::computeBdW(real3 *BdW, cudaStream_t st){
      sys->log<System::DEBUG3>("[BDHI::Cholesky] BdW");  
      if(!isMup2date) setup_step();
      /*computeBdw messes up M, fills it with B*/
      isMup2date = false;

      int numberParticles = pg->getNumberParticles();

      
      CusolverSafeCall(cusolverDnSetStream(solver_handle, st));

      real * d_M = thrust::raw_pointer_cast(mobilityMatrix.data());
      /*Perform cholesky factorization, store B on LOWER part of M matrix*/

      CusolverSafeCall(cusolverDnpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
				       3*numberParticles, d_M, 3*numberParticles, d_work, h_work_size, d_info));

      curandSetStream(curng, st);

      /*Gen new noise in BdW*/
      curandGenerateNormal(curng,
			   (real*) BdW,
			   3*numberParticles + ((3*numberParticles)%2),
			   real(0.0), real(1.0));

      CublasSafeCall(cublasSetStream(handle, st));

      /*Compute B·dW -> y = M·y*/
      CublasSafeCall(cublastrmv(handle, //B is an upper triangular matrix (with non unit diagonal)
		 CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
		 3*numberParticles,
		 d_M, 3*numberParticles,
				  (real*)BdW, 1));
      

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
    
	inline __device__ real3 zero(){return make_real3(real(0.0));}
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
      Cholesky_ns::divMTransverser divMtr(divM, selfMobility, par.hydrodynamicRadius);
      NBody nbody_divM(pd, pg, sys);
  
      nbody_divM.transverse(divMtr, st);
    }
  }
}
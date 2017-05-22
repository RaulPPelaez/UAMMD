/*Raul P. Pelaez 2017. Lanczos algorithm


TODO:
100- Change name of function to solveResult or something other than noise
100- gemv is probably taking too much in the GPU for small iter, measure and use cblas below a threshold


 */
#include"LanczosAlgorithm.cuh" 



#ifdef USE_MKL
 #include<mkl.h>
#else
 #include<lapacke.h>
 #ifdef USE_CPU_LAPACK_AND_BLAS
  #include<cblas.h>
 #endif
#endif

#include"lapack_and_blas_defines.h"


#include<fstream>

void LanczosAlgorithm::init(){
  if(this->initialized) return;
  /*Init cuSolver and cuBLAS for Lanczos process*/
  status = cublasCreate(&cublas_handle);
  if(status){cerr<<"ERROR with CUBLAS!!\n"<<endl; exit(1);}
  cusolverDnCreate(&solver_handle);
  h_work_size = 0;//work size of operation  
  cusolverDngesvd_bufferSize(solver_handle, max_iter, max_iter, &h_work_size);

  if(!d_work)
    gpuErrchk(cudaMalloc(&d_work, h_work_size));
  if(!d_info)
    gpuErrchk(cudaMalloc(&d_info, sizeof(int)));

  /*Create noise*/
  curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(curng, grng.next());
  /*Create a temporal vector to warm up curand*/
  GPUVector3 noise(N);
  //Curand fill with gaussian numbers with mean 0 and var 1
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(curng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  this->initialized = true;
}

LanczosAlgorithm::LanczosAlgorithm(int N, real tolerance): 
  N(N),
  h_work_size(0), d_work(nullptr), d_info(nullptr),
  max_iter(3),  min_iter(3), check_convergence_steps(3), tolerance(tolerance),
  w(N),
  V(3*N*min_iter),
  P(min_iter*min_iter),
  hdiag(min_iter), hsup(min_iter), htemp(2*min_iter),
  old_noise(N), initialized(false)
{
  //this->init();

  old_noise.memset(0);
}

void LanczosAlgorithm::increment_max_iter(){
  real inc = 2;
  GPUVector<real> tmpGPU(3*N*max_iter);
  real tmpCPU[max_iter*max_iter];

  cudaMemcpy(tmpGPU, V.d_m, 3*N*max_iter*sizeof(real), cudaMemcpyDeviceToDevice);    
  V = GPUVector<real>(3*N*(max_iter+inc));
  cudaMemcpy(V, tmpGPU, 3*N*max_iter*sizeof(real), cudaMemcpyDeviceToDevice);

  P = Vector<real>((max_iter+inc)*(max_iter+inc));

  fori(0,max_iter) tmpCPU[i] = hdiag[i];
  hdiag = Vector<real>((max_iter+inc));
  fori(0,max_iter) hdiag[i] = tmpCPU[i];

  fori(0,max_iter) tmpCPU[i] = hsup[i];
  hsup = Vector<real>((max_iter+inc));
  fori(0,max_iter) hsup[i] = tmpCPU[i];
  
  htemp = Vector<real>(2*(max_iter+inc));
     
  max_iter += inc;


  h_work_size = 0;//work size of operation  
  cusolverDngesvd_bufferSize(solver_handle, max_iter, max_iter, &h_work_size);

  cudaFree(d_work);
  gpuErrchk(cudaMalloc(&d_work, h_work_size));
  
}

void LanczosAlgorithm::compNoise(real z2, int N, int iter, real * BdW, cudaStream_t st){
  
  real alpha = 1.0;
  real beta = 0.0;
  /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/
  /**** H^1/2·e1 = Pt· first_column_of(sqrt(Hdiag)·P) ******/
  
  /**************LAPACKE********************/

  /*The tridiagonal matrix is stored only with its diagonal and subdiagonal*/
  /*Store both in a temporal array*/
  fori(0,iter){
    htemp[i] = hdiag[i];
    htemp[i+iter]= hsup[i];
  }
  /*P = eigenvectors must be filled with zeros, I do not know why*/
  memset(P.data, 0, iter*iter*sizeof(real));
  /*Compute eigenvalues and eigenvectors of a triangular symmetric matrix*/
  auto info = LAPACKE_steqr(LAPACK_COL_MAJOR, 'i',
			    iter, htemp.data, htemp.data+iter,
			    P.data, iter);

  /***Hdiag_temp = Hdiag·P·e1****/
  forj(0,iter){
    htemp[j] = sqrt(htemp[j])*P[iter*j];
  }
   

  /***** Htemp = H^1/2·e1 = Pt· hdiag_temp ****/
  /*Compute with blas*/
#ifdef USE_CPU_LAPACK_AND_BLAS
  real *temp = htemp.data+iter;   
  cblas_gemv(CblasColMajor, CblasNoTrans,
	     iter, iter,
	     alpha,
	     P.data, iter,
	     htemp.data, 1,
	     beta,
	     temp, 1);
  cudaMemcpy(htemp.d_m, temp, iter*sizeof(real), cudaMemcpyHostToDevice);
   
#else
  /*Compute with cublas*/
  htemp.upload();  
  P.upload();
  real *temp = htemp.d_m+iter;
  cublasSetStream(cublas_handle, st);
  auto status = cublasgemv(cublas_handle,
   			   CUBLAS_OP_N,
   			   iter, iter,
   			   &alpha,
   			   P.d_m, iter,
   			   htemp.d_m, 1,
   			   &beta,
   			   temp, 1);
   
  cudaMemcpyAsync(htemp, temp, iter*sizeof(real), cudaMemcpyDeviceToDevice, st);
#endif
  cublasSetStream(cublas_handle, st);
  /*noise = ||z||_2 * Vm · H^1/2 · e1 = Vm · (z2·hdiag_temp)*/
  beta = 0.0;
  cublasgemv(cublas_handle, CUBLAS_OP_N,
	     3*N, iter,
	     &z2,
	     V, 3*N,
	     htemp.d_m, 1,
	     &beta,
	     BdW, 1);  
}


/*Raul P. Pelaez 2016. Computes BdW, see .h*/

#include"BrownianNoiseBDHI.h"
#include<fstream>
#include "misc/Diagonalize.h"

using namespace brownian_hy_euler_maruyama_ns;


/*Definition of the BrownianNoise computer base class, handles a cuRand array*/
BrownianNoiseComputer::BrownianNoiseComputer(uint N): N(N), noise(N +((3*N)%2)), noiseTemp(nullptr){
  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, grng.next());
  //Curand fill with gaussian numbers with mean 0 and var 1    
  noise.fill_with(make_real3(real(0.0)));
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));   
}

BrownianNoiseComputer::~BrownianNoiseComputer(){
  curandDestroyGenerator(rng);
  if(noiseTemp)
    cudaFree(noiseTemp);
}

real* BrownianNoiseComputer::genNoiseNormal(real mean, real std){
  /*Gen new noise*/
  if(!noiseTemp){
    cudaMalloc(&noiseTemp, (3*N+((3*N)%2))*sizeof(real));
  }
  curandGenerateNormal(rng, noiseTemp,  3*N + ((3*N)%2), mean, std);  
  return noiseTemp;
}


/***************************CHOLESKY***************************/

bool BrownianNoiseCholesky::init(Diffusion &D, uint N){
  cerr<<"\tInitializing Cholesky subsystem...";
  cusolverDnCreate(&solver_handle);
  h_work_size = 0;//work size of operation

  if(!D.getMatrix()){
    cerr<<"\t\tERROR: Cant use cholesky with a Matrix-Free method!!"<<endl;
    exit(1);
  }
  /*Initialize cusolver*/

  cusolverDnpotrf_bufferSize(solver_handle, 
			     CUBLAS_FILL_MODE_UPPER, 3*N, D.getMatrix()->d_m, 3*N, &h_work_size);
  gpuErrchk(cudaMalloc(&d_work, h_work_size*sizeof(real)));
  gpuErrchk(cudaMalloc(&d_info, sizeof(int)));
  cerr<<"DONE!!"<<endl;
  return true;
}


real* BrownianNoiseCholesky::compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream){
  cusolverDnSetStream(solver_handle, stream);
  /*Perform cholesky factorization, store B on LOWER part of D matrix*/
  cusolverDnpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		  3*N, D.getMatrix()->d_m, 3*N, d_work, h_work_size, d_info);
  /*Gen new noise*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  /*Print dw*/
  // noise.download();
  // static ofstream out("noise.dat");
  // fori(0,3*N)
  //   out<<((real*)(noise.data))[i]<<endl;
  
  cudaStreamSynchronize(stream);
  /*Compute B·dW, store in noise*/
  cublastrmv(handle,
	     CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
	     3*N,
	     D.getMatrix()->d_m, 3*N,
	     (real*)noise.d_m, 1);

  /*Print B*/
  // Matrixf* d= D.getMatrix();
  // d->download();
  // ofstream out3("B.dat");
  // for(int i=0; i<3*N; i++){
  //   for(int j=0; j<3*N; j++){
  //     //if(i<j) out3<<0<<" ";
  //     //else
  // 	 out3<< d[0][i][j]<<" ";
  //   }
  //   out3<<endl;
  // }
  //  exit(0);

  /*Print BdW*/
  // noise.download();
  // static ofstream out2("noiseChol.dat");
  // fori(0,3*N)
  //   out2<<((real*)(noise.data))[i]<<endl;
  // exit(0);

  return (real*)noise.d_m;
}
/*********************************LANCZOS**********************************/


BrownianNoiseLanczos::BrownianNoiseLanczos(uint N, uint max_iter):
  BrownianNoiseComputer(N), max_iter(max_iter), w(N),
  V(max_iter, 3*N), H(max_iter, max_iter), Htemp(max_iter, max_iter),
  P(max_iter, max_iter),   Pt(max_iter, max_iter),
  hdiag(max_iter), hdiag_temp(max_iter),  hsup(max_iter),
  h_work_size(0),
  d_work(nullptr),  d_info(nullptr)
{}

bool BrownianNoiseLanczos::init(Diffusion &D, uint N){
  cerr<<"\tInitializing Lanczos subsystem...";
  w.fill_with(make_real3(0));  w.upload();
  w.download();
  V.fill_with(0);  V.upload();

  H.fill_with(0);  H.upload();  

  hdiag.fill_with(0);  hdiag.upload();

  hsup.fill_with(0);  hsup.upload();

  /*Initialize cusolver to diagonalize and cublas for matrix algebra*/
  cusolverDnCreate(&solver_handle);
  cublasCreate(&cublas_handle);
  cusolverDngesvd_bufferSize(solver_handle, max_iter, max_iter, &h_work_size);

  
  gpuErrchk(cudaMalloc(&d_work, h_work_size));
  gpuErrchk(cudaMalloc(&d_info, sizeof(int)));

  cerr<<"DONE!!"<<endl;
  return true;
}


real* BrownianNoiseLanczos::compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream){
  /*See J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347*/
  // Vector3 noise_prev(N);
  // noise_prev.fill_with(make_real3(0));
  /************v[0] = z/||z||_2*****/
  /*Compute noise*/       /*V.d_m -> first column of V*/
  curandGenerateNormal(rng, V.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));

  /*Print dw*/
  // V.download(3*N);
  // ofstream nout("noise.dat");
  // fori(0,3*N) nout<<V.data[i]<<endl;
  
  /*1/norm(z)*/
  real invz2; cublasnrm2(handle, 3*N, V.d_m, 1, &invz2); invz2 = 1.0/invz2;

  /*v[0] = v[0]*1/norm(z)*/ 
  cublasscal(handle, 3*N, &invz2,  V.d_m, 1);
  
  real alpha=1.0;
  /*Lanczos iterations for Krylov decomposition*/
  int i;
  for(i=0; i<max_iter; i++){
    /*w = D·vi*/
    D.dot(V.d_m+3*N*i, (real*)w.d_m, handle);
    
    if(i>0){
      /*w = w-h[i-1][i]·vi*/
      alpha = -hsup[i-1];
      cublasaxpy(handle, 3*N,
		 &alpha,
		 V.d_m+3*N*(i-1), 1,
		 (real *)w.d_m, 1);
    }

    /*h[i][i] = dot(w, vi)*/
    cublasdot(handle, 3*N,
	      (real *)w.d_m, 1,
	      V.d_m+3*N*i, 1,
	      &(hdiag[i]));
    if(i<(int)max_iter-1){
      /*w = w-h[i][i]·vi*/
      alpha = -hdiag[i];
      cublasaxpy(handle, 3*N,
		 &alpha,
		 V.d_m+3*N*i, 1,
		 (real *)w.d_m, 1);
      /*h[i+1][i] = h[i][i+1] = norm(w)*/
      cublasnrm2(handle, 3*N, (real*)w.d_m, 1, &(hsup[i]));
      /*v_(i+1) = w·1/ norm(w)*/
      if(hsup[i]>real(0.0)){
	real invw2 = 1.0/hsup[i];
	cublasscal(handle, 3*N, &invw2,  (real *)w.d_m, 1);
      }
      else{/*If norm(w) = 0 that means all elements of w are zero, so set the first to 1*/
	real one = 1;
	cudaMemcpy(w.d_m, &one , sizeof(real), cudaMemcpyHostToDevice);
      }
      cudaMemcpy(V.d_m+3*N*(i+1), w.d_m, 3*N*sizeof(real), cudaMemcpyDeviceToDevice);
    }
  }
  
  /*Compute BdW using h and dw*/
  this->compNoise(1.0/invz2, N, i-1);
  
  return (real*)noise.d_m;
}
  
/*Computes the brownian noise in the current Lanczos iteration,
  z2 is the norm of the initial random vector,
  iter is the current iteration*/
void BrownianNoiseLanczos::compNoise(real z2, uint N, uint iter){
  real alpha = 1.0;
  real beta = 0.0;

  fori(0,iter) hdiag_temp[i] = hdiag[i];
  
  /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/


  /*Compute H^1/2 by diagonalization***********/
  /* H^1/2 = P · Hdiag^1/2 · P^T */
  /******WITH CUSOLVER SVD****************/
  fori(0,iter*iter) H.data[i] = 0;
  fori(0,iter){
    H.data[i*iter+i] = hdiag_temp[i];
    if((uint)i<iter-1){
      H.data[(i+1)*iter+i] = hsup[i];
    }
    if(i>0){
      H.data[(i-1)*iter+i] = hsup[i-1];
    }
  }

  H.upload(iter*iter);
  /*Diagonalize H*/
  cudaDeviceSynchronize();
  cusolverDngesvd(solver_handle, 'A', 'A', iter, iter,
		  H.d_m, iter,
		  hdiag_temp.d_m,
		  P.d_m, iter,
		  Pt.d_m, iter,
		  d_work, h_work_size, nullptr ,d_info);  
  cudaDeviceSynchronize();
  
  hdiag_temp.download(iter);
  
  /*sqrt H*/
  fori(0, iter){
    if(hdiag_temp[i]>0)
      hdiag_temp[i] = sqrt(hdiag_temp[i]);
    else hdiag_temp[i] = 0.0;
  }    
  /* store in hdiag_temp -> H^1/2·e1 = P·Hdiag_Temp^1/2·Pt·e1*/
  fori(0,iter*iter) H.data[i] = 0;
  fori(0,iter) H.data[i*iter+i] = hdiag_temp[i];
  
  H.upload(iter*iter);  
  cublasgemm(cublas_handle,
    	     CUBLAS_OP_N, CUBLAS_OP_N,
    	     iter,iter,iter,
    	     &alpha,
    	     H, iter,
    	     Pt, iter,
    	     &beta,
    	     Htemp,  iter);
  cublasgemm(cublas_handle,
    	     CUBLAS_OP_N,  CUBLAS_OP_N,
    	     iter, iter, iter,
    	     &alpha,
    	     P, iter,
    	     Htemp, iter,
    	     &beta,
   	     H, iter);
  /*Now H contains H^1/2*/  
  /*Store H^1/2 · e1 in hdiag_temp*/
  /*hdiag_temp = H^1/2·e1*/
  cudaMemcpy(hdiag_temp.d_m, H.d_m, iter*sizeof(real), cudaMemcpyDeviceToDevice);
  
  /*noise = ||z||_2 * Vm · H^1/2 · e1 = Vm · (z2·hdiag_temp)*/
  cublasStatus_t st = cublasgemv(cublas_handle, CUBLAS_OP_N,
				 3*N, iter,
				 &z2,
				 V, 3*N,
				 hdiag_temp.d_m, 1,
				 &beta,
				 (real*)noise.d_m, 1);
  /*Print BdW*/
  // noise.download();
  //  ofstream out("noiseLanc.dat");
  //  fori(0,3*N)
  //    out<<((real*)noise.data)[i]<<endl;
  //  exit(0);

}

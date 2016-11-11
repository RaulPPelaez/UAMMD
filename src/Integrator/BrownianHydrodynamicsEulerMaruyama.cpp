/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation
  
  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+D·F[t]) + sqrt(2dt)·dW·B
   Being:
     X - Positions
     D - Diffusion matrix
     K - Shear matrix
     dW- Noise vector, gaussian with mean 0 and var 1
     B - sqrt(D)

  Similar to Brownian Euler Maruyama, but now the Diffusion matrix has size 3Nx3N and is updated
    each step according to the Rotne Prager method.

TODO:
100- Optimize
100- Use a 3x3 K matrix
100- Do not store the entire 3Nx3N D matrix, update each particle indepently and update positions, computing B along the way
*/


#include "BrownianHydrodynamicsEulerMaruyama.h"
//#include <cub/cub.cuh>

using namespace brownian_hy_euler_maruyama_ns;

BrownianHydrodynamicsEulerMaruyama::BrownianHydrodynamicsEulerMaruyama(Matrixf D0in, Matrixf Kin,
								       StochasticNoiseMethod stochMethod):
  Integrator(),
  force3(N),
  DF(N),
  D(3*N, 3*N){

  cerr<<"Initializing Brownian Dynamics with Hydrodynamics (Euler Maruyama)..."<<endl;

  this->K = Kin;
  this->D0 = D0in;
  if(!D0.isSym()){
    cerr<<"D0 Matrix must be symmetric!!"<<endl;
    exit(1);
  }
  if(D0.n != 9 || K.n !=9){
    cerr<<"D0 and K must be 3x3!!"<<endl;
    exit(1);
  }
  
  params.sqrtdt = sqrt(dt);
  params.dt = dt;
  params.D0 = D0[0][0];
  params.rh = 0.5;
  params.N = N;
  params.L = L;
  
  cudaStreamCreate(&stream);
  cudaStreamCreate(&stream2);

  switch(stochMethod){
  case(CHOLESKY):
    cuBNoise = make_shared<BrownianNoiseCholesky>(N);
    break;
  case(LANCZOS):
    cuBNoise = make_shared<BrownianNoiseLanczos>(N);
    break;
    
  }
  
  
  status = cublasCreate(&handle);

  /*The 3x3 shear matrix is encoded as an array of 3 real3*/
  K.upload();

  
  uint i,l, k;
  D.fill_with(0.0f);

  /* Diffusion tensor (diagonal boxes remain unchanged during execution) */
  for(i = 0; i < N; i++){
    for(k = 0; k < 3; k++){
      for(l = 0; l < 3; l++){
	D[3*i + k][3*i + l] = D0[k][l];
      }
    }
  }
  D.upload();

  DF.fill_with(make_real3(0.0)); DF.upload();

  //  cuChol.init(D, N);
  cuBNoise->init(D,N);
  
  initGPU(params);
  
  cerr<<"Brownian Dynamics with Hydrodynamics (Euler Maruyama)\t\tDONE!!\n\n";
}
BrownianHydrodynamicsEulerMaruyama::~BrownianHydrodynamicsEulerMaruyama(){
  gpuErrchk(cudaStreamDestroy(stream));
  gpuErrchk(cudaStreamDestroy(stream2));
  cublasDestroy(handle);
  //cusolverSpDestroy(solver_handle);
}

void BrownianHydrodynamicsEulerMaruyama::update(){
  steps++; 
  real alpha = real(1.0), beta = real(0.0);
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";

  /*Reset force*/
  cudaMemset((real *)force.d_m, 0, N*sizeof(real4));

  /*Compute new force*/
  for(auto forceComp: interactors) forceComp->sumForce();

  /*Update D according to the positions*/
  rodne_callGPU(D, pos, stream, N);
  
  /*Copy force array into a real3 array to multiply by D using cublas*/
  real4_to_real3GPU(force, force3, N);

    
  /*Generate new noise*/
  //curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  
  alpha = dt;

  // cublasSgemv(handle, CUBLAS_OP_T,
  // 	      3*N, 3*N,
  // 	      &alpha,
  // 	      K.d_m, 3*N,
  // 	      (real*)(pos3.d_m), 1,
  // 	      &beta,
  // 	      (real*) KR.d_m, 1);

  cudaStreamSynchronize(stream);
  cublasSetStream(handle, stream);
  /*Compute D·F*/
  cublassymv(handle, CUBLAS_FILL_MODE_UPPER,
	      3*N, 
		&alpha,
		D.d_m, 3*N,
		(real*)(force3.d_m), 1,
		&beta,
		(real*) DF.d_m, 1);
  cudaStreamSynchronize(stream);
  
  /*Compute the brownian Noise array BdW*/
  //cuChol.compute(D, N, stream);
  real *BdW = cuBNoise->compute(handle, D, N);
     
  cudaDeviceSynchronize();
  /*Update the positions*/
  /* R += KR +DF + BdW*/ 
  integrateGPU(pos, DF.d_m, (real3*)BdW, (real3 *)K.d_m, N);
}

real BrownianHydrodynamicsEulerMaruyama::sumEnergy(){
  return real(0.0);
}


/*******BROWNIAN NOISE BASE CLASS**************/

/*Definition of the BrownianNoise computer base class, handles a cuRand array*/
BrownianNoiseComputer::BrownianNoiseComputer(uint N): noise(N +((3*N)%2)){
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
}






/***************************CHOLESKY***************************/

bool BrownianNoiseCholesky::init(real *D, uint N){
  cusolverDnCreate(&solver_handle);
  h_work_size = 0;//work size of operation
    
  cusolverDnpotrf_bufferSize(solver_handle, 
			     CUBLAS_FILL_MODE_UPPER, 3*N, D, 3*N, &h_work_size);
  gpuErrchk(cudaMalloc(&d_work, h_work_size*sizeof(real)));
  gpuErrchk(cudaMalloc(&d_info, sizeof(int)));
  
  return true;
}


real* BrownianNoiseCholesky::compute(cublasHandle_t handle, real *D, uint N, cudaStream_t stream){
  cusolverDnSetStream(solver_handle, stream);
  cusolverDnpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		  3*N, D, 3*N, d_work, h_work_size, d_info);
  
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));

  cudaStreamSynchronize(stream);
  cublastrmv(handle,
	     CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
	     3*N,
	     D, 3*N,
	     (real*)noise.d_m, 1);

  // int m_info;
  // cudaMemcpy(&m_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  // cerr<<" "<<m_info<<endl;
  return (real*)noise.d_m;
}
/*********************************LANCZOS**********************************/


BrownianNoiseLanczos::BrownianNoiseLanczos(uint N, uint max_iter):
  BrownianNoiseComputer(N), w(N),
  V(3*N, max_iter), H(max_iter, max_iter),
  hdiag(max_iter), hsup(max_iter)
  //  ,cub_storage(nullptr),cub_storage_size(0)
{
}

bool BrownianNoiseLanczos::init(real *D, uint N){

  w.fill_with(make_real3(0));  w.upload();
  V.fill_with(0);  V.upload();

  H.fill_with(0);  H.upload();  

  hdiag.fill_with(0);  hdiag.upload();

  hsup.fill_with(0);  hsup.upload();

  
  /*Compute cub storage needed*/
  // cub::DeviceReduce::Sum(cub_storage, cub_storage_size,
  // 			 (real*)noise.d_m, w.d_m /*dummy pointer*/, 3*N);
  // cerr<<"\tStorage needed by cub: "<<cub_storage_size<<" bytes"<<endl;
  // cudaMalloc(&cub_storage, cub_storage_size); 
  
  return true;
}


real* BrownianNoiseLanczos::compute(cublasHandle_t handle, real *D, uint N, cudaStream_t stream){

  real *d_temp;
  cudaMalloc(&d_temp, sizeof(real));
  /************v[0] = z/||z||_2*****/
  /*Compute noise*/       /*V.d_m -> first column of V*/  
  curandGenerateNormal(rng, V.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  
  /*norm(z)*/
  cublasnrm2(handle, 3*N, V.d_m, 1, d_temp);
  /*v[0] = v[0]*1/norm(z)*/ 
  //cublasscal(
  
  

  cudaFree(d_temp);
  return (real*)noise.d_m;
}












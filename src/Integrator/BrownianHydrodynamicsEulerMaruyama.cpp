/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
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
100- Do not store the entire 3Nx3N D matrix, update each particle indepently and update positions
*/


#include "BrownianHydrodynamicsEulerMaruyama.h"


BrownianHydrodynamicsEulerMaruyama::BrownianHydrodynamicsEulerMaruyama():
  Integrator(),
  noise(N +((3*N)%2)), pos3(N), force3(N),
  K(3*N, 3*N), D(3*N, 3*N){

  cerr<<"Initializing Brownian Dynamics with Hydrodynamics (Euler Maruyama)..."<<endl;
  
  params.sqrtdt = sqrt(dt);

  cudaStreamCreate(&stream);
  cudaStreamCreate(&stream2);
  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, 1234ULL);
  
  //Curand fill with gaussian numbers with mean 0 and var 1    
  noise.fill_with(make_float3(0.0f));
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  
  status = cublasCreate(&handle);
  
  float shear = 5.0f;
  K.fill_with(0.0f);
  fori(0,N){
    K[3*i][3*i+1] = shear; 
  }
  K.upload();

  int i,l, k;
  D.fill_with(0.0f);
  /* Diffusion tensor (diagonal boxes remain unchanged during execution) */
  for(i = 0; i < N; i++){
    for(k = 0; k < 3; k++){
      for(l = 0; l < 3; l++){
        if(k == l) D[3*i + k][3*i + l] = 1;
        else D[3*i + k][3*i + l] = 0;
      }
    }
  }
  D.upload();
  
  KR = Vector<float3>(N);KR.fill_with  (make_float3(0.0f)); KR.upload();
  DF = Vector<float3>(N);DF.fill_with  (make_float3(0.0f)); DF.upload();

  initBrownianHydrodynamicsEulerMaruyamaGPU(params);
  
  cerr<<"Brownian Dynamics with Hydrodynamics (Euler Maruyama)\t\tDONE!!\n\n";
}
BrownianHydrodynamicsEulerMaruyama::~BrownianHydrodynamicsEulerMaruyama(){}

void BrownianHydrodynamicsEulerMaruyama::update(){
  steps++; 
  float alpha = 1.0f, beta = 0.0f;
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";
  
  cudaMemset((float *)force->d_m, 0.0f, 4*N*sizeof(float));
  
  for(auto forceComp: interactors) forceComp->sumForce();
  
  copy_pos(pos->d_m, pos3, force->d_m, force3, N);

  
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  
  rodne_call(D, pos3, stream, N);

  cublasSetStream(handle, stream2);
  alpha = dt;
  cublasSgemv(handle, CUBLAS_OP_T,
	      3*N, 3*N,
	      &alpha,
	      K.d_m, 3*N,
	      (float*)(pos3.d_m), 1,
	      &beta,
	      (float*) KR.d_m, 1);

  
  cublasSetStream(handle, stream);
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER,
   	      3*N, 
   	      &alpha,
   	      D.d_m, 3*N,
   	      (float*)(force3.d_m), 1,
   	      &beta,
   	      (float*) DF.d_m, 1);
  cudaStreamSynchronize(stream);
  chol();
  cudaStreamSynchronize(stream);
  //No BdW array needed!
  cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
	      3*N,
	      D.d_m, 3*N,
	      (float*)noise.d_m, 1);
  
  cudaDeviceSynchronize();
  integrateBrownianHydrodynamicsEulerMaruyamaGPU(pos->d_m, DF.d_m, noise.d_m, KR.d_m, dt, N);
}

float BrownianHydrodynamicsEulerMaruyama::sumEnergy(){
  return 0.0f;
}
void BrownianHydrodynamicsEulerMaruyama::chol(){
  static bool first_time = true;
  static int h_work_size;
  static float *d_work;

  if(first_time){
    cusolverDnCreate(&solver_handle);
    h_work_size = 0;//work size of operation
    cusolverDnSpotrf_bufferSize(solver_handle, 
				CUBLAS_FILL_MODE_UPPER, 3*N, D.d_m, 3*N, &h_work_size);
    cudaMalloc(&d_work, h_work_size*sizeof(float));
    first_time = false;
  }
  /*Query the cholesky decomposition*/
  int *d_info;

  cusolverDnSetStream(solver_handle, stream);
  cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		   3*N, D.d_m, 3*N, d_work, h_work_size, d_info);
  
}



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


BrownianHydrodynamicsEulerMaruyama::BrownianHydrodynamicsEulerMaruyama():
  Integrator(),
  force3(N),
  noise(N +((3*N)%2)), DF(N),
  D(3*N, 3*N), K(4, 4){

  cerr<<"Initializing Brownian Dynamics with Hydrodynamics (Euler Maruyama)..."<<endl;
  
  params.sqrtdt = sqrt(dt);
  params.dt = dt;
  params.D0 = 1.0f;
  params.rh = 1.0f;
  
  cudaStreamCreate(&stream);
  cudaStreamCreate(&stream2);
  
  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, gcnf.seed);
  
  //Curand fill with gaussian numbers with mean 0 and var 1    
  noise.fill_with(make_float3(0.0f));
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  
  status = cublasCreate(&handle);
  
  float shear = 5.0f;
  K.fill_with(0.0f);
  K[0][1] = shear; 
  K.upload();
  /*The 4x4 shear matrix is encoded as an array of 4 float4*/
  /*It is 4x4 instead of 3x3 for optimization*/
  params.K = (float4*)K.d_m;

  
  uint i,l, k;
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
  
  DF.fill_with  (make_float3(0.0f)); DF.upload();

  initBrownianHydrodynamicsEulerMaruyamaGPU(params);
  
  cerr<<"Brownian Dynamics with Hydrodynamics (Euler Maruyama)\t\tDONE!!\n\n";
}
BrownianHydrodynamicsEulerMaruyama::~BrownianHydrodynamicsEulerMaruyama(){}

void BrownianHydrodynamicsEulerMaruyama::update(){
  steps++; 
  float alpha = 1.0f, beta = 0.0f;
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";

  /*Reset force*/
  cudaMemset((float *)force.d_m, 0.0f, 4*N*sizeof(float));

  /*Compute new force*/
  for(auto forceComp: interactors) forceComp->sumForce();

  /*Update D according to the positions*/
  rodne_call(D, pos, stream, N);
  
  /*Copy force array into a float3 array to multiply by D using cublas*/
  copy_pos(force, force3, N);

    
  /*Generate new noise*/
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  
  alpha = dt;
  
  //cublasSetStream(handle, stream2);
  // cublasSgemv(handle, CUBLAS_OP_T,
  // 	      3*N, 3*N,
  // 	      &alpha,
  // 	      K.d_m, 3*N,
  // 	      (float*)(pos3.d_m), 1,
  // 	      &beta,
  // 	      (float*) KR.d_m, 1);

  cudaStreamSynchronize(stream);
  cublasSetStream(handle, stream);
  /*Compute D·F*/
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER,
   	      3*N, 
   	      &alpha,
   	      D.d_m, 3*N,
   	      (float*)(force3.d_m), 1,
   	      &beta,
   	      (float*) DF.d_m, 1);
  cudaStreamSynchronize(stream);
  /*Perform a cholesky decomposition on D to obtain B, 
    stored in the same matrix*/
  chol();
  cudaStreamSynchronize(stream);
  /*Compute the random force*/
  //No BdW array needed!
  /*By overwritting the diagonal of D we can store B and D in the same matrix*/
  //B is encoded in the upper part of the D Matrix*/
  //The diagonal part of D is fixedin rodne call*/
  cublasStrmv(handle,
	      CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
	      3*N,
	      D.d_m, 3*N,
	      (float*)noise.d_m, 1);
  
  cudaDeviceSynchronize();
  /*Update the positions*/
  integrateBrownianHydrodynamicsEulerMaruyamaGPU(pos, DF.d_m, noise.d_m, N);
}

float BrownianHydrodynamicsEulerMaruyama::sumEnergy(){
  return 0.0f;
}
void BrownianHydrodynamicsEulerMaruyama::chol(){
  static bool first_time = true;
  static int h_work_size;
  static float *d_work;
  static int *d_info;

  if(first_time){
    cusolverDnCreate(&solver_handle);
    h_work_size = 0;//work size of operation
    cusolverDnSpotrf_bufferSize(solver_handle, 
				CUBLAS_FILL_MODE_UPPER, 3*N, D.d_m, 3*N, &h_work_size);
    cudaMalloc(&d_work, h_work_size*sizeof(float));
    cudaMalloc(&d_info, sizeof(int));
    first_time = false;
  }
  /*Query the cholesky decomposition*/

  cusolverDnSetStream(solver_handle, stream);
  cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		   3*N, D.d_m, 3*N, d_work, h_work_size, d_info);
  // int m_info;
  // cudaMemcpy(&m_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  // cerr<<" "<<m_info<<endl;
  
}



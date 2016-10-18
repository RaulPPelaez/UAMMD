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


using namespace brownian_hy_euler_maruyama_ns;

BrownianHydrodynamicsEulerMaruyama::BrownianHydrodynamicsEulerMaruyama(Matrixf D0, Matrixf K):
  Integrator(),
  force3(N),
  noise(N +((3*N)%2)), DF(N), BdW(3*N),
  D(3*N, 3*N), K(K), D0(D0){

  cerr<<"Initializing Brownian Dynamics with Hydrodynamics (Euler Maruyama)..."<<endl;

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
  
  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, grng.next());
  
  //Curand fill with gaussian numbers with mean 0 and var 1    
  noise.fill_with(make_real3(real(0.0)));
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  BdW.fill_with(real(0.0));
  BdW.upload();
 
  
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

  cuChol.init(D, N);
  
  initGPU(params);
  
  cerr<<"Brownian Dynamics with Hydrodynamics (Euler Maruyama)\t\tDONE!!\n\n";
}
BrownianHydrodynamicsEulerMaruyama::~BrownianHydrodynamicsEulerMaruyama(){
  curandDestroyGenerator(rng);
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
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  
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
  /*Perform a cholesky decomposition on D to obtain B, 
    stored in the same matrix*/  
  cuChol.compute(D, N, stream);
  
  //Code here will execute concurrently
  
  
  cudaStreamSynchronize(stream);
  /*Compute the random force*/
  //No BdW array needed!
  /*By overwritting the diagonal of D we can store B and D in the same matrix
    B is encoded in the upper part of the D Matrix
    The diagonal part of D is fixedin rodne call*/
  /* B · dW */
  cublastrmv(handle,
	     CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
	     3*N,
	     D.d_m, 3*N,
	     (real*)noise.d_m, 1);
  
  //trmul(D,(real *) noise.d_m, BdW, 3*N); 
  cudaDeviceSynchronize();
  /*Update the positions*/
  /* R += KR +DF + BdW*/ 
  integrateGPU(pos, DF.d_m, (real3*)noise.d_m, (real3 *)K.d_m, N);
}

real BrownianHydrodynamicsEulerMaruyama::sumEnergy(){
  return real(0.0);
}




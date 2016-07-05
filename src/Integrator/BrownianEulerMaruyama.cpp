/*Raul P. Pelaez 2016. Brownian Euler Maruyama Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+D·F[t]) + sqrt(dt)·dW·B
   Being:
     X - Positions
     D - Diffusion matrix
     K - Shear matrix
     dW- Noise vector
     B - sqrt(D)
*/
#include "BrownianEulerMaruyama.h"



void cholesky(Vector<float4> Din, Vector<float4> &Bout){
  int i, j, k; /* Indices */
  float tmpsum; /* Temporary variable */

  Matrix<float> B(3,3);
  Matrix<float> D(3,3);

  fori(0,3){
    D[i][0] = Din[i].x;
    D[i][1] = Din[i].y;
    D[i][2] = Din[i].z;
  }

  /* Clear B matrix */
  B.fill_with(0.0f);
  Bout.fill_with(make_float4(0.0f));



  for(j = 0; j < 3; j++) {
    tmpsum = 0;
    for(k = 0; k < j; k++)
      tmpsum += B[j][k]*B[j][k];
    B[j][j] = sqrt(D[j][j] - tmpsum);

    for(i = j + 1; i < 3; i++) {
      tmpsum = 0;
      for(k = 0; k < j; k++)
        tmpsum += B[i][k]*B[j][k];
      B[i][j] = (D[i][j] - tmpsum)/B[j][j];
    }
  }

  fori(0,3){
    Bout[i].x = B[i][0];
    Bout[i].y = B[i][1];
    Bout[i].z = B[i][2];
  }
}

BrownianEulerMaruyama::BrownianEulerMaruyama(Vector4Ptr D,
					     Vector4Ptr K):
  Integrator(),
  D(D),K(K),
  noise(N){

  
  params.sqrtdt = sqrt(dt)*sqrt(2.0f);

  D->upload();
  params.D = D->d_m;
  K->upload();
  params.K = K->d_m;

  B = Vector<float4>(4);

  B.fill_with(make_float4(0.0f));
  cholesky(*D, B);
  B.upload();
  params.B = B.d_m;
  
  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, time(NULL));

  noise.fill_with(make_float3(0.0f));
  noise.upload();
  //Curand fill with gaussian numbers with mean 0 and var 1
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N, 0.0f, 1.0f);
  
  initBrownianEulerMaruyamaGPU(params);
}
BrownianEulerMaruyama::~BrownianEulerMaruyama(){}

void BrownianEulerMaruyama::update(){
  steps++;
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";

  curandGenerateNormal(rng, (float*) noise.d_m, 3*N, 0.0f, 1.0f);
  
  cudaMemset((float *)force->d_m, 0.0f, 4*N*sizeof(float));
  for(auto forceComp: interactors) forceComp->sumForce();

  integrateBrownianEulerMaruyamaGPU(pos->d_m, noise, force->d_m, dt, N);
}

float BrownianEulerMaruyama::sumEnergy(){
  return 0.0f;
}

/*Raul P. Pelaez 2016. TwoStepVelVerlet Integrator derived class

  Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of creating the velocities and keep the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
  
  TODO:
   Maybe the velocities should be outside the module, handled as the positions.

 */
#include "BrownianHydrodynamicsEulerMaruyama.h"


BrownianHydrodynamicsEulerMaruyama::BrownianHydrodynamicsEulerMaruyama(shared_ptr<Vector<float4>> pos,
								       shared_ptr<Vector<float4>> force,
								       uint N, float L, float dt):
  noise(N),
  Integrator(pos, force, N, L, dt){

  params.sqrtdt = sqrt(dt);


  cudaStreamCreate(&stream);
  cudaStreamCreate(&stream2);
  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, rand());

  noise.fill_with(make_float3(0.0f));
  noise.upload();
  //Curand fill with gaussian numbers with mean 0 and var 1
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N, 0.0f ,1.0f);

  float shear = 0.0f;
  K = Matrix<float3>(3*N,3*N);
  K.fill_with(0.0f);
  fori(0,N){
    K[3*i][3*i+1] = shear; 
  }
  K.upload();


  D = Matrix<float>(3*N,3*N);
  /* Diffusion tensor (diagonal boxes remain unchanged during execution) */
  for(i = 0; i < N; i++){
    for(k = 0; k < 3; k++){
      for(l = 0; l < 3; l++){
        if(k == l) D[3*i + k][3*i + l] = D0;
        else D[3*i + k][3*i + l] = 0;
      }
    }
  } 
  D.upload();


  B = Matrix<float>(3*N, 3*N);
  B.fill_with(0.0f);
  B.upload();


  KR = Vector<float3>(N);KR.fill_with(0.0f); KR.upload();
  DF = Vector<float3>(N);DF.fill_with(0.0f); DF.upload();
  BdW = Vector<float3>(N);BdW.fill_with(0.0f); BdW.upload();

  initBrownianHydrodynamicsEulerMaruyamaGPU(params);

}
BrownianHydrodynamicsEulerMaruyama::~BrownianHydrodynamicsEulerMaruyama(){}

void BrownianHydrodynamicsEulerMaruyama::update(){
  steps++; 

  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";

  curandGenerateNormal(rng, (float*) noise.d_m, 3*N, 0.0f, 1.0f);
  for(auto forceComp: interactors) forceComp->sumForce();

  rodne_prage(stream);

  MVprod(K.d_m, (float*)pos.d_m, (float*) KR.d_m, dt, 0.0f);

  cudaStreamSynchronize(stream);

  chol();

  MVprod(D.d_m, (float *)force.d_m, (float*) DF.d_m, dt, 0.0); /* D·F·dt */
  MVprod(B, (float*)noise.d_m, (float*) BdW, sqrt(2)*sqrt(dt), 0.0); 

  integrateBrownianHydrodynamicsEulerMaruyamaGPU(pos->d_m, DF->d_m, BdW.d_m, KR.d_m, dt, N);
  

}




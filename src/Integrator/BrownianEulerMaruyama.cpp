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


BrownianEulerMaruyama::BrownianEulerMaruyama(shared_ptr<Vector<float4>> pos,
					     shared_ptr<Vector<float4>> force,
					     shared_ptr<Vector<float4>> D,
					     shared_ptr<Vector<float4>> K,
					     uint N, float L, float dt):
  D(D),K(K),
  noise(N),
  Integrator(pos, force, N, L, dt){

  params.sqrtdt = sqrt(dt);

  D->upload();
  params.D = D->d_m;
  K->upload();
  params.K = K->d_m;

  //  B = D;
  B->fill_with(make_float4(0.0f));
  B->upload();
  params.B = B->d_m;

  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, rand());

  noise.fill_with(make_float3(0.0f));
  noise.upload();
  //Curand fill with gaussian numbers with mean 0 and var 1
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N, 0.0f ,1.0f);

  initBrownianEulerMaruyamaGPU(params);

}
BrownianEulerMaruyama::~BrownianEulerMaruyama(){}

void BrownianEulerMaruyama::update(){
    steps++;
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";

  curandGenerateNormal(rng, (float*) noise.d_m, 3*N, 0.0f, 1.0f);
  for(auto forceComp: interactors) forceComp->sumForce();

  integrateBrownianEulerMaruyamaGPU(pos->d_m, noise, force->d_m, dt, N);

}

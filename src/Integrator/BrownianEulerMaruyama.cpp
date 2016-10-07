/*Raul P. Pelaez 2016. Brownian Euler Maruyama Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided as a global object,
     they are not created by the module.
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


using namespace brownian_euler_maruyama_ns;
/*Performs the cholesky decomposition of a matrix, in CPU*/
Matrixf cholesky(Matrixf Din){
  //Doesnt check for positive definite
  //Super slow, use only in initialization
  uint i, j, k; /* Indices */
  float tmpsum; /* Temporary variable */
  if(!Din.isSquare()){
    cerr<<"Cholesky: Not a square matrix!!!"<<endl;
  }
  uint dim = Din.size().x;

  Matrixf B(dim, dim);
  Matrixf D = Din;
  /* Clear B matrix */
  B.fill_with(0.0f);
  
  for(j = 0; j < dim; j++) {
    tmpsum = 0;
    for(k = 0; k < j; k++)
      tmpsum += B[j][k]*B[j][k];
    B[j][j] = sqrt(D[j][j] - tmpsum);

    for(i = j + 1; i < dim; i++) {
      tmpsum = 0;
      for(k = 0; k < j; k++)
        tmpsum += B[i][k]*B[j][k];
      B[i][j] = (D[i][j] - tmpsum)/B[j][j];
    }
  }
  
  return B;
}

BrownianEulerMaruyama::BrownianEulerMaruyama(Matrixf Din,
					     Matrixf Kin):
  Integrator(),
  D(Din), K(Kin),
  noise( N + ((3*N)%2) ){
  
  cerr<<"Initializing Brownian Euler Maruyama Integrator..."<<endl;
  if(Din.size().x!=3 || Kin.size().x!=3 ||
     !Din.isSquare() || !Kin.isSquare()){
    cerr<<"ERROR!, K and D must be 3x3!!"<<endl;
    exit(1);
  }

  /*Set GPU parameters*/
  
  params.sqrtdt = sqrt(dt)*sqrt(2.0f);
  params.dt = dt;
  params.L = L;
  params.N = N;

  D.upload();
  K.upload();
  params.D = (float3 *)D.d_m;
  params.K = (float3* )K.d_m;

  B = cholesky(D);
  B.upload();
  params.B = (float3* )B.d_m;

  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, grng.next());

  noise.fill_with(make_float3(0.0f));
  noise.upload();
  //Curand fill with gaussian numbers with mean 0 and var 1, you have to ask for an even number of them
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  
  initGPU(params);
  cerr<<"Brownian Euler Maruyama Integrator\t\tDONE!!\n\n"<<endl;
}
BrownianEulerMaruyama::~BrownianEulerMaruyama(){}

void BrownianEulerMaruyama::update(){
  steps++;
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";
  /*Generate noise*/
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  /*Reset force*/
  cudaMemset(force.d_m, 0.0f, N*sizeof(float4));
  /*Compute new forces*/
  for(auto forceComp: interactors) forceComp->sumForce();
   /*Update positions*/
  integrateGPU(pos, noise, force, N);
}

float BrownianEulerMaruyama::sumEnergy(){
  return 0.0f;
}

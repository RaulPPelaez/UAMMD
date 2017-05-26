/*Raul P. Pelaez 2016. Brownian Euler Maruyama Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided as a global object,
     they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2·T·dt)·dW·B
   Being:
     X - Positions
     M - Mobility matrix
     K - Shear matrix
     dW- Noise vector
     B - chol(M)
*/
#include "BrownianEulerMaruyama.cuh"
#include"utils/vector_overloads.h"
#include"utils/helper_gpu.cuh"
__constant__ real3 Mcons[3];
__constant__ real3 Bcons[3];
__constant__ real3 Kcons[3];


/*Performs the cholesky decomposition of a matrix, in CPU*/
Matrixf cholesky(Matrixf Din){
  //Doesnt check for positive definite
  //Super slow, use only in initialization
  uint i, j, k; /* Indices */
  real tmpsum; /* Temporary variable */
  if(!Din.isSquare()){
    cerr<<"Cholesky: Not a square matrix!!!"<<endl;
  }
  uint dim = Din.size().x;

  Matrixf B(dim, dim);
  Matrixf D = Din;
  /* Clear B matrix */
  B.fill_with(0.0);
  
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



BrownianEulerMaruyama::BrownianEulerMaruyama(Matrixf Min,
					     Matrixf Kin):
  BrownianEulerMaruyama(Min, Kin, gcnf.N, gcnf.L, gcnf.dt){}
BrownianEulerMaruyama::BrownianEulerMaruyama(Matrixf Min,
					     Matrixf Kin,
					     int N, real3 L, real dt):
  Integrator(N, L, dt, 128),
  M(Min), K(Kin),
  noise( N + ((3*N)%2) ),
  T(gcnf.T){
  
  cerr<<"Initializing Brownian Euler Maruyama Integrator..."<<endl;
  if(Min.size().x!=3 || Kin.size().x!=3 ||
     !Min.isSquare() || !Kin.isSquare()){
    cerr<<"ERROR!, K and D must be 3x3!!"<<endl;
    exit(1);
  }

  /*Set GPU parameters*/
  this->sqrt2Tdt = sqrt(dt)*sqrt(2.0*T);
  
  M.upload();
  K.upload();

  //Is K zero?
  bool K0 = true;
  fori(0,9){
    if(K.data[i] != real(0.0)) K0 = false;
  }
  if(K0){
    K.freeMem();
  }
  
  B = cholesky(M);
  B.upload();

  cudaMemcpyToSymbol(Mcons, M.d_m, 3*sizeof(real3));
  cudaMemcpyToSymbol(Bcons, B.d_m, 3*sizeof(real3));
  if(!K0)
    cudaMemcpyToSymbol(Kcons, K.d_m, 3*sizeof(real3));
  
  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, grng.next());

  noise.fill_with(make_real3(0.0));
  noise.upload();
  //Curand fill with gaussian numbers with mean 0 and var 1, you have to ask for an even number of them
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), 0.0, 1.0);
  
  cerr<<"Brownian Euler Maruyama Integrator\t\tDONE!!\n\n"<<endl;
}
BrownianEulerMaruyama::~BrownianEulerMaruyama(){}

namespace BrownianEulerMaruyama_ns{
  /*Integrate the movement*/
  template<bool shear>
  __global__ void integrateGPU(real4 __restrict__  *pos,
			       const real4 __restrict__  *force,
			       const real3 __restrict__ *dW,
			       // const real3 __restrict__ *M,
			       // const real3 __restrict__ *B,
			       // const real3 __restrict__ *K,
			       int N, real dt, bool D2, real sqrt2Tdt){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    /*Half step velocity*/

    real3 *M = Mcons;
    real3 *B = Bcons;
    
    real3 p = make_real3(pos[i]);
    real3 f = make_real3(force[i]);

    real3 KR = make_real3(real(0.0));
    if(shear){
      real3 *K = Kcons;
      KR = make_real3(dot(K[0],p), dot(K[1],p), dot(K[2],p));
    }
    
    // X[t+dt] = dt(K·X[t]+D·F[t]) + sqrt(dt)·dW·B
    p.x = dt*(  KR.x + dot(M[0],f)) + sqrt2Tdt*dot(dW[i],B[0]);
    p.y = dt*(  KR.y + dot(M[1],f)) + sqrt2Tdt*dot(dW[i],B[1]);
    if(!D2)//If 3D
      p.z =  dt*( KR.z + dot(M[2],f)) + sqrt2Tdt*dot(dW[i],B[2]);

    pos[i] += make_real4(p);
  }

};
void BrownianEulerMaruyama::update(){
  uint nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0); 

  steps++;
  /*Generate noise*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));
  /*Reset force*/
  cudaMemset(force.d_m, 0, N*sizeof(real4));
  /*Compute new forces*/
  for(auto forceComp: interactors) forceComp->sumForce();
   /*Update positions*/
  if(K.d_m)
  BrownianEulerMaruyama_ns::integrateGPU<true><<<nblocks,nthreads>>>(pos.d_m, force.d_m, noise.d_m,
		  //			     (real3*)M.d_m, (real3*)B.d_m, (real3*)K.d_m,
				     N, dt, gcnf.D2, sqrt2Tdt);
  else
    BrownianEulerMaruyama_ns::integrateGPU<false><<<nblocks,nthreads>>>(pos.d_m, force.d_m, noise.d_m,
								       N, dt, gcnf.D2, sqrt2Tdt);

  
}

real BrownianEulerMaruyama::sumEnergy(){
  return 0.0;
}

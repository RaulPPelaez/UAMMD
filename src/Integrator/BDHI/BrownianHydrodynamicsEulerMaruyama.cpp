//WARNING: DIFFUSION REFERS TO MOBILITY M = D/kT
/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation
   
  Solves the following stochastich differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*kb*T*dt)·B·dW
   Being:
     X - Positions
     M - Mobility matrix -> M = D/kT
     K - Shear matrix
     dW- Brownian noise vector
     B - B*B^T = M -> i.e Cholesky decomposition B=chol(M) or Square root B=sqrt(M)

  The Diffusion matrix is computed via the Rotne Prager Yamakawa tensor

  The module offers several ways to compute and sovle the different terms.

  The brownian Noise can be computed by:
     -Computing B·dW explicitly performing a Cholesky decomposition on M.
     -Through a Lanczos iterative method to reduce M to a smaller Krylov subspace and performing the operation there.

  On the other hand the mobility(diffusion) matrix can be handled in several ways:
     -Storing and computing it explicitly as a 3Nx3N matrix.
     -Not storing it and recomputing it when a product M·v is needed.

REFERENCES:

1- Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
        J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347

TODO:
100- Optimize
100- Change Diffusion to Mobility, M = D/kT.  to avoid the diveregence when T=0. Basically just change the names.
80-  Put D0 and K in constant memory.
*/


#include "BrownianHydrodynamicsEulerMaruyama.h"


using namespace brownian_hy_euler_maruyama_ns;

BrownianHydrodynamicsEulerMaruyama::BrownianHydrodynamicsEulerMaruyama(Matrixf D0in, Matrixf Kin,
								       StochasticNoiseMethod stochMethod,
								       DiffusionMatrixMode matrixMode, int max_iter):
  Integrator(),
  force3(N),
  DF(N), divM(N)
  {  

  cerr<<"Initializing Brownian Dynamics with Hydrodynamics (Euler Maruyama)..."<<endl;

  this->K = Kin;
  if(K.n !=9){
    cerr<<"K must be 3x3!!"<<endl;
    exit(1);
  }
  
  params.sqrt2Tdt = sqrt(2*dt*gcnf.T);
  params.T = gcnf.T;
  params.invDelta = 1.0/dt; //For RFD
  params.dt = dt;
  params.N = N;
  params.L = L;

  cerr<<"\tTemperature: "<<gcnf.T<<endl;
  
  cudaStreamCreate(&stream);
  cudaStreamCreate(&stream2);

  switch(matrixMode){
  case MATRIXFULL:
  case DEFAULT:
    D = make_shared<DiffusionFullMatrix>(D0in, N);
    break;
  case MATRIXFREE:
    D = make_shared<DiffusionMatrixFree>(D0in, N);
    break;   
  }
  

  
  switch(stochMethod){
  case(CHOLESKY):
    cuBNoise = make_shared<BrownianNoiseCholesky>(N);
    break;
  case(LANCZOS):
    if(max_iter == 0){
      max_iter = 5;
      //Rule of thumb for iterations
      if(N>500) max_iter = 7;
      if(N>5000) max_iter = 10;
      if(N>10000) max_iter = 20;
      if(N>100000) max_iter = 30;
    }
    cuBNoise = make_shared<BrownianNoiseLanczos>(N, max_iter);
    break;
    
  }
  
  
  status = cublasCreate(&handle);

  /*The 3x3 shear matrix is encoded as an array of 3 real3, should be in constant memory*/
  K.upload();
  
  /*Result of multiplyinf D·F*/
  DF.fill_with(make_real3(0.0)); DF.upload();
  divM.fill_with(make_real3(0.0)); divM.upload();
  /*Init brownian noise generator*/
  cuBNoise->init(*D,N);
  
  initGPU(params);

  cerr<<"Brownian Dynamics with Hydrodynamics (Euler Maruyama)\t\tDONE!!\n\n";
}
BrownianHydrodynamicsEulerMaruyama::~BrownianHydrodynamicsEulerMaruyama(){
  cerr<<"Destroying BrownianHydrodynamicsEulerMarujama...";
  gpuErrchk(cudaStreamDestroy(stream));
  gpuErrchk(cudaStreamDestroy(stream2));
  cublasDestroy(handle);
  cerr<<"DONE!!"<<endl;
  //cusolverSpDestroy(solver_handle);
}

/*Advance the simulation one time step*/
void BrownianHydrodynamicsEulerMaruyama::update(){
  steps++;


  /*Reset force*/
  cudaMemset((real *)force.d_m, 0, N*sizeof(real4));

  /*Compute new force*/
  for(auto forceComp: interactors) forceComp->sumForce();

  /*Copy force array into a real3 array to multiply by D using cublas*/
  real4_to_real3GPU(force, force3, N);
  
  /*Update D according to the positions*/
  /*If this is a Matrix free method Diffusion::compute just does nothing*/
  D->compute();

     
  cudaDeviceSynchronize();
  /*Compute DF = D·F, can be done concurrently with cuBNoise*/
  D->dot((real*) (force3.d_m), (real*)DF.d_m, handle, stream);
    
  /*Compute the brownian Noise array BdW*/
  cudaDeviceSynchronize();
   real3 *BdW = nullptr;
   if(gcnf.T>0)
     BdW = (real3*) cuBNoise->compute(handle, *D, N, stream);   
   
  /* divM =  (M(q+dw)-M(q))·dw/d^2*/
   if(L.z == real(0.0)){
     //real* kk =  (real*)cuBNoise->genNoiseNormal(0.0f, 1.0f/params.invDelta);
     D->divergence((real*) divM.d_m, nullptr);
   }
  cudaDeviceSynchronize();
  
  /*Update the positions*/
  /* R += KR + MF + sqrt(2dtT)BdW + kTdivM*/ 
  integrateGPU(pos, DF, (real3*)(BdW), divM, (real3 *)K.d_m, N);
}

real BrownianHydrodynamicsEulerMaruyama::sumEnergy(){
  return real(0.0);
}



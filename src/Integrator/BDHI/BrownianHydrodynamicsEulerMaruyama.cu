/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation
   
  Solves the following stochastich differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*kb*T*dt)·B·dW + T·divM·dt(in 2D)
   Being:
     X - Positions
     M - Mobility matrix -> M = D/kT
     K - Shear matrix
     dW- Brownian noise vector
     B - B*B^T = M -> i.e Cholesky decomposition B=chol(M) or Square root B=sqrt(M)

  The Mobility matrix is computed via the Rotne Prager Yamakawa tensor.

  The module offers several ways to compute and solve the different terms.
  
     BDHI::Cholesky:
     -Computing M·F and B·dW  explicitly storing M and performing a Cholesky decomposition on M.

     BDHI::Lanczos:
     -A Lanczos iterative method to reduce M to a smaller Krylov subspace and performing the operation B·dW there, the product M·F is performed in a matrix-free way, recomputing M every time M·v is needed.

     BDHI::PSE:
     -The Positively Split Edwald Method, which takes the computation to fourier space.

REFERENCES:

1- Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
        J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347
2- Rapid sampling of stochastic displacements in Brownian dynamics simulations 
        The Journal of Chemical Physics 146, 124116 (2017); doi: http://dx.doi.org/10.1063/1.4978242

TODO:
100- Optimize streams
*/


#include "BrownianHydrodynamicsEulerMaruyama.h"
#include<fstream>
using namespace std;


BrownianHydrodynamicsEulerMaruyama::BrownianHydrodynamicsEulerMaruyama(Matrixf Kin, real vis, real rh,
								       BDHIMethod bdhiMethod,
								       real tolerance):
  Integrator(),
  MF(N), BdW(N+1)
{  
  BLOCKSIZE = 1024; /*threads per block*/
  cerr<<"Initializing Brownian Dynamics with Hydrodynamics (Euler Maruyama)..."<<endl;
  
  if(gcnf.D2) divM = GPUVector3(N);
  
  this->K = Kin;
  if(K.n !=9){
    cerr<<"K must be 3x3!!"<<endl;
    exit(1);
  }
  /*The 3x3 shear matrix is encoded as an array of 3 real3, should be in constant memory*/
  K.upload();
  
  nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  nblocks = N/nthreads +  ((N%nthreads!=0)?1:0);
   
  real M0 = 1.0/(6*M_PI*vis*rh);
  cerr<<"\tTemperature: "<<gcnf.T<<endl;
  cerr<<"\tM0: "<<M0<<endl;
  cerr<<"\tvis: "<<vis<<endl;
  cerr<<"\trh: "<<rh<<endl;
  
  cudaStreamCreate(&stream);
  cudaStreamCreate(&stream2);



  
  real psi = 1.2;
  switch(bdhiMethod){
  case CHOLESKY:
    M0 = 1/(6*M_PI*vis*rh);    
    bdhi = make_shared<BDHI::Cholesky>(M0, rh, N);
    break;
  case DEFAULT:
  case PSE:
    bdhi = make_shared<BDHI::PSE>(vis, gcnf.T, rh, psi, N);
    break;
  case LANCZOS:
    M0 = 1/(6*M_PI*vis*rh);
    bdhi = make_shared<BDHI::Lanczos>(M0, rh, N, tolerance);
    break;   
  }
  
  /*Result of multiplyinf M·F*/
  MF.fill_with(make_real3(0.0));
  BdW.fill_with(make_real3(0.0));


  cerr<<"Brownian Dynamics with Hydrodynamics (Euler Maruyama)\t\tDONE!!\n\n";
}
BrownianHydrodynamicsEulerMaruyama::~BrownianHydrodynamicsEulerMaruyama(){
  cerr<<"Destroying BrownianHydrodynamicsEulerMarujama...";
  cerr<<"DONE!!"<<endl;
}




namespace BDHI_EulerMaruyama_ns{
  /*
    dR = dt(KR+MF) + sqrt(2*T*dt)·BdW +T·divM·dt
  */
  /*With all the terms computed, update the positions*/
  /*T=0 case is templated*/
  template<bool noise>
  __global__ void integrateGPUD(real4* __restrict__ pos,
				const real3* __restrict__ MF,
				const real3* __restrict__ BdW,
				const real3* __restrict__ K,
				const real3* __restrict__ divM, int N,
				real sqrt2Tdt, real T, real dt, bool D2){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;    
    if(i>=N) return;
    /*Position and color*/
    real4 pc = pos[i];
    real3 p = make_real3(pc);
    real c = pc.w;

    /*Shear stress*/
   real3 KR = make_real3(0);
    KR.x = dot(K[0], p);
    KR.y = dot(K[1], p);
    /*2D clause. Although K[2] should be 0 in 2D anyway...*/
    if(!D2)
      KR.z = dot(K[2], p);
    
    /*Update the position*/
    p += (KR + MF[i])*dt;
    /*T=0 is treated specially, there is no need to produce noise*/
    if(noise){
      real3 bdw  = BdW[i];
      if(D2)
	bdw.z = 0;
      p += sqrt2Tdt*bdw;
    }
    /*If we are in 2D and the divergence term exists*/
    if(divM){
      real3 divm = divM[i];
      //divm.z = real(0.0);
      //p += params.T*divm*params.invDelta*params.invDelta*params.dt; //For RFD
      p += T*dt*divm;
    }           
    /*Write to global memory*/
    pos[i] = make_real4(p,c);
  }
}


/*Advance the simulation one time step*/
void BrownianHydrodynamicsEulerMaruyama::update(){

  /*
    dR = dt(KR+MF) + sqrt(2*T*dt)·BdW +T·divM·dt
  */
  steps++;
  /*Reset force*/
  cudaMemset((real *)force.d_m, 0, N*sizeof(real4));

  /*Compute new force*/
  for(auto forceComp: interactors) forceComp->sumForce();

  bdhi->setup_step();

  bdhi->computeMF(MF.d_m, stream);
  
  bdhi->computeBdW(BdW, stream);
  /* divM =  (M(q+dw)-M(q))·dw/d^2*/
  if(gcnf.D2){
    bdhi->computeDivM(divM.d_m, stream2);
  }

  real sqrt2Tdt = sqrt(2*dt*gcnf.T);

  bdhi->finish_step(stream);
  /*Update the positions*/
  /* R += KR + MF + sqrt(2dtT)BdW + kTdivM*/
  cudaDeviceSynchronize();
  if(gcnf.T > 0)
    BDHI_EulerMaruyama_ns::integrateGPUD<true><<<nblocks, nthreads>>>(pos,
					      MF, (real3*)(BdW), (real3 *)K.d_m, divM.d_m,
					      N, sqrt2Tdt, gcnf.T, dt, gcnf.D2);  
  else
    BDHI_EulerMaruyama_ns::integrateGPUD<false><<<nblocks, nthreads>>>(pos,
					       MF, (real3*)(BdW), (real3 *)K.d_m, divM.d_m,
					       N, sqrt2Tdt, gcnf.T, dt, gcnf.D2);
  
}

real BrownianHydrodynamicsEulerMaruyama::sumEnergy(){
  return real(0.0);
}

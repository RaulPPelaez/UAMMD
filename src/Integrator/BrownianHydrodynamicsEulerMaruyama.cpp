/*Raul P. Pelaez 2016. Brownian Euler Maruyama with hydrodynamics Integrator derived class implementation
   
  Solves the following stochastich differential equation:
      X[t+dt] = dt(K·X[t]+D·F[t]) + sqrt(dt)·B·dW
   Being:
     X - Positions
     D - Diffusion matrix
     K - Shear matrix
     dW- Brownian noise vector
     B - sqrt(D)

 The Diffusion matrix is computed via the Rotne Prager Yamakawa tensor

 The module offers several ways to compute and sovle the different terms.

 The brownian Noise can be computed by:
     -Computing sqrt(D)·dW explicitly performing a Cholesky decomposition on D.
     -Through a Lanczos iterative method to reduce D to a smaller Krylov subspace and performing the operation there.

  On the other hand the mobility(diffusion) matrix can be handled in several ways:
     -Storing and computing it explicitly as a 3Nx3N matrix.
     -Not storing it and recomputing it when a product D·v is needed.

REFERENCES:

1- Krylov subspace methods for computing hydrodynamic interactions in Brownian dynamics simulations
        J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347

TODO:
100- Optimize
80-  Put D0 and K in constant memory, and fully use them
*/


#include "BrownianHydrodynamicsEulerMaruyama.h"
#include "misc/Diagonalize.h"

using namespace brownian_hy_euler_maruyama_ns;

BrownianHydrodynamicsEulerMaruyama::BrownianHydrodynamicsEulerMaruyama(Matrixf D0in, Matrixf Kin,
								       StochasticNoiseMethod stochMethod,
								       DiffusionMatrixMode mode, int max_iter):
  Integrator(),
  force3(N),
  DF(N),
  D(D0in, N, (stochMethod==CHOLESKY)?MATRIXFULL:mode){  

  cerr<<"Initializing Brownian Dynamics with Hydrodynamics (Euler Maruyama)..."<<endl;

  this->K = Kin;
  if(K.n !=9){
    cerr<<"K must be 3x3!!"<<endl;
    exit(1);
  }
  
  params.sqrtdt = sqrt(dt);
  params.dt = dt;
  params.N = N;
  params.L = L;
  
  cudaStreamCreate(&stream);
  cudaStreamCreate(&stream2);

  switch(stochMethod){
  case(CHOLESKY):
    cuBNoise = make_shared<BrownianNoiseCholesky>(N);
    break;
  case(LANCZOS):
    if(max_iter == 0){
      max_iter = 5;
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

  /*Init brownian noise generator*/
  cuBNoise->init(D,N);
  
  initGPU(params);

  cerr<<"Brownian Dynamics with Hydrodynamics (Euler Maruyama)\t\tDONE!!\n\n";
}
BrownianHydrodynamicsEulerMaruyama::~BrownianHydrodynamicsEulerMaruyama(){
  gpuErrchk(cudaStreamDestroy(stream));
  gpuErrchk(cudaStreamDestroy(stream2));
  cublasDestroy(handle);
  //cusolverSpDestroy(solver_handle);
}

/*Advance the simulation one time step*/
void BrownianHydrodynamicsEulerMaruyama::update(){
  steps++;
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";

  /*Reset force*/
  cudaMemset((real *)force.d_m, 0, N*sizeof(real4));

  /*Compute new force*/
  for(auto forceComp: interactors) forceComp->sumForce();

  /*Update D according to the positions*/
  /*If this is a Matrix free method Diffusion::compute just does nothing*/
  D.compute();
  
  /*Copy force array into a real3 array to multiply by D using cublas*/
  real4_to_real3GPU(force, force3, N);
   
  cudaStreamSynchronize(stream);
  cublasSetStream(handle, stream);
  
  /*Compute DF = D·F, can be done concurrently with cuBNoise*/
  D.dot((real*) (force3.d_m), (real*)DF.d_m, handle);
    
  cublasSetStream(handle, 0);
  /*Compute the brownian Noise array BdW*/
  real *BdW = cuBNoise->compute(handle, D, N);
     
  cudaDeviceSynchronize();
  /*Update the positions*/
  /* R += KR +DF + BdW*/ 
  integrateGPU(pos, DF.d_m, (real3*)BdW, (real3 *)K.d_m, N);
}

real BrownianHydrodynamicsEulerMaruyama::sumEnergy(){
  return real(0.0);
}
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*********************************DIFFUSION HANDLER****************************/


/*Diffusion constructor*/
Diffusion::Diffusion(Matrixf D0in, uint N, DiffusionMatrixMode mode):
  N(N), /*Number of particles, not degrees of freedom*/
  mode(mode==DEFAULT?MATRIXFULL:mode){


  
  /*Store the matrix only in FULL mode*/
  D = Matrixf(this->mode==MATRIXFULL?(3*N):1,this->mode==MATRIXFULL?(3*N):1);
  
    this->D0 = D0in;
    if(!D0.isSym()){
      cerr<<"D0 Matrix must be symmetric!!"<<endl;
      exit(1);
    }
    if(D0.n != 9){
      cerr<<"D0 must be 3x3!!"<<endl;
      exit(1);
    }     


    D.fill_with(0.0f);

    // /* Diffusion tensor (diagonal boxes remain unchanged during execution) */
    // uint i,l, k;
    // for(i = 0; i < N; i++){
    //   for(k = 0; k < 3; k++){
    // 	for(l = 0; l < 3; l++){
    // 	  D[3*i + k][3*i + l] = D0[k][l];
    // 	}
    //   }
    // }
    D.upload();
    
    params.D0 = D0[0][0];/*All D0 should be in constant memory, not just the first element*/    
    params.rh = 0.5*gcnf.sigma;
    initRPYGPU(params);
}

void Diffusion::compute(){
  if(mode==MATRIXFULL){
    /*Computes only UPPER part of D*/
    computeDiffusionRPYGPU(D, pos, 0, N);
    
    // D.download();
    // ofstream out("D.dat");
    // for(int i=0; i<3*N; i++){
    //   for(int j=0; j<3*N; j++)
    //     if(i<j)
    // 	 out<<D[j][i]<<" ";
    //     else 	out<< D[i][j]<<" ";
    //   out<<endl;
    // }
  }
  else if(mode == MATRIXFREE){
    /*Does nothing*/
  }

}
/*Computes Dv = D·v, it is not needed to pass handle in a matrix free method*/
void Diffusion::dot(real *v, real *Dv, cublasHandle_t handle){

  if(mode==MATRIXFULL){
    if(handle==0){
      cerr<<"ERROR: Diffusion::dot: I need a cublas handle in a MATRIX FULL Diffusion method!!!"<<endl;
      exit(1);
    }
      
    real alpha = 1.0;
    real beta = 0;
    /*Compute D·v*/
    cublassymv(handle, CUBLAS_FILL_MODE_UPPER,
	       3*N, 
	       &alpha,
	       D.d_m, 3*N,
	       v, 1,
	       &beta,
	       Dv, 1);
  }
  else if(mode == MATRIXFREE){
    diffusionDotGPU(pos, (real3*)v, (real3*)Dv, N);
  }
}

real* Diffusion::getMatrix(){
  switch(mode){
  case MATRIXFULL:
    return this->D.d_m;
    break;
  case MATRIXFREE:
    cerr<<"\tERROR: There is no Matrix in a matrix free Diffusion mode!!"<<endl;
    exit(1);
    break;
  }
}

/*******BROWNIAN NOISE BASE CLASS**************/

/*Definition of the BrownianNoise computer base class, handles a cuRand array*/
BrownianNoiseComputer::BrownianNoiseComputer(uint N): noise(N +((3*N)%2)){
  /*Create noise*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, grng.next());
  //Curand fill with gaussian numbers with mean 0 and var 1    
  noise.fill_with(make_real3(real(0.0)));
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));   
}

BrownianNoiseComputer::~BrownianNoiseComputer(){
  curandDestroyGenerator(rng);
}






/***************************CHOLESKY***************************/

bool BrownianNoiseCholesky::init(Diffusion &D, uint N){
  cerr<<"\tInitializing Cholesky subsystem...";
  cusolverDnCreate(&solver_handle);
  h_work_size = 0;//work size of operation

  if(!D.getMatrix()){
    cerr<<"\t\tERROR: Cant use cholesky with a Matrix-Free method!!"<<endl;
    exit(1);
  }
  /*Initialize cusolver*/
  cusolverDnpotrf_bufferSize(solver_handle, 
			     CUBLAS_FILL_MODE_UPPER, 3*N, D.getMatrix(), 3*N, &h_work_size);
  gpuErrchk(cudaMalloc(&d_work, h_work_size*sizeof(real)));
  gpuErrchk(cudaMalloc(&d_info, sizeof(int)));
  cerr<<"DONE!!"<<endl;
  return true;
}


real* BrownianNoiseCholesky::compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream){
  cusolverDnSetStream(solver_handle, stream);
  /*Perform cholesky factorization, store B on LOWER part of D matrix*/
  cusolverDnpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		  3*N, D.getMatrix(), 3*N, d_work, h_work_size, d_info);

  /*Gen new noise*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));

  cudaStreamSynchronize(stream);
  /*Compute B·dW, store in noise*/
  cublastrmv(handle,
	     CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
	     3*N,
	     D.getMatrix(), 3*N,
	     (real*)noise.d_m, 1);

  // noise.download();
  // static ofstream out("noiseChol.dat");
  // fori(0,N)
  //   out<<noise[i]<<endl;

  // int m_info;
  // cudaMemcpy(&m_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  // cerr<<" "<<m_info<<endl;
  return (real*)noise.d_m;
}
/*********************************LANCZOS**********************************/


BrownianNoiseLanczos::BrownianNoiseLanczos(uint N, uint max_iter):
  BrownianNoiseComputer(N), max_iter(max_iter), w(N),
  V(max_iter, 3*N), H(max_iter, max_iter), Htemp(max_iter, max_iter),
  P(max_iter, max_iter),   Pt(max_iter, max_iter),
  hdiag(max_iter), hdiag_temp(max_iter),  hsup(max_iter),
  h_work_size(0),
  d_work(nullptr),  d_info(nullptr)
{
}

bool BrownianNoiseLanczos::init(Diffusion &D, uint N){
  cerr<<"\tInitializing Lanczos subsystem...";
  w.fill_with(make_real3(0));  w.upload();
  w.download();
  V.fill_with(0);  V.upload();

  H.fill_with(0);  H.upload();  

  hdiag.fill_with(0);  hdiag.upload();

  hsup.fill_with(0);  hsup.upload();

  /*Inbitialize cusolver to diagonalize/cholesky and cublas for matrix algebra*/
  cusolverDnCreate(&solver_handle);
  cublasCreate(&cublas_handle);
  cusolverDngesvd_bufferSize(solver_handle, max_iter, max_iter, &h_work_size);
  // cusolverDnpotrf_bufferSize(solver_handle, 
  // 			     CUBLAS_FILL_MODE_UPPER, max_iter, H.d_m, max_iter, &h_work_size);

  
  gpuErrchk(cudaMalloc(&d_work, h_work_size));
  gpuErrchk(cudaMalloc(&d_info, sizeof(int)));

  cerr<<"DONE!!"<<endl;
  return true;
}


real* BrownianNoiseLanczos::compute(cublasHandle_t handle, Diffusion &D, uint N, cudaStream_t stream){
  /*See J. Chem. Phys. 137, 064106 (2012); doi: 10.1063/1.4742347*/
  // real *d_temp;
  // cudaMalloc(&d_temp, sizeof(real));
  Vector3 noise_prev(N);
  noise_prev.fill_with(make_real3(0));
  /************v[0] = z/||z||_2*****/
  /*Compute noise*/       /*V.d_m -> first column of V*/
  curandGenerateNormal(rng, V.d_m, 3*N + ((3*N)%2), real(0.0), real(1.0));

  // V.download();
  // ofstream nout("noise.dat");
  // fori(0,3*N) nout<<V.data[i]<<endl;
  
  /*1/norm(z)*/
  real invz2; cublasnrm2(handle, 3*N, V.d_m, 1, &invz2); invz2 = 1.0/invz2;

  /*v[0] = v[0]*1/norm(z)*/ 
  cublasscal(handle, 3*N, &invz2,  V.d_m, 1);
  
  real alpha=1.0;
  /*Lanczos iterations for Krylov decomposition*/
  int i;
  for(i=0; i<max_iter; i++){
    /*w = D·vi*/
    D.dot(V.d_m+3*N*i, (real*)w.d_m, handle);
    
    if(i>0){
      /*w = w-h[i-1][i]·vi*/
      alpha = -hsup[i-1];
      cublasaxpy(handle, 3*N,
		 &alpha,
		 V.d_m+3*N*(i-1), 1,
		 (real *)w.d_m, 1);
    }

    /*h[i][i] = dot(w, vi)*/
    cublasdot(handle, 3*N,
	      (real *)w.d_m, 1,
	      V.d_m+3*N*i, 1,
	      &(hdiag[i]));
    if(i<(int)max_iter-1){
      /*w = w-h[i][i]·vi*/
      alpha = -hdiag[i];
      cublasaxpy(handle, 3*N,
		 &alpha,
		 V.d_m+3*N*i, 1,
		 (real *)w.d_m, 1);
      /*h[i+1][i] = h[i][i+1] = norm(w)*/
      cublasnrm2(handle, 3*N, (real*)w.d_m, 1, &(hsup[i]));
      /*v_(i+1) = w·1/ norm(w)*/
      if(hsup[i]>real(0.0)){
	real invw2 = 1.0/hsup[i];
	cublasscal(handle, 3*N, &invw2,  (real *)w.d_m, 1);
      }
      else{/*If norm(w) = 0 that means all elements of w are zero, so set the first to 1*/
	real one = 1;
	cudaMemcpy(w.d_m, &one , sizeof(real), cudaMemcpyHostToDevice);
      }
      cudaMemcpy(V.d_m+3*N*(i+1), w.d_m, 3*N*sizeof(real), cudaMemcpyDeviceToDevice);
    }

    // if(i>3){
      
    //   this->compNoise(1.0/invz2, N, i);

    //   noise.download();
    //   real noiseSum = 0, noiseMax = 0;
    //   float *n=(float*)&noise[0], *np = (float*)&noise_prev[0];
    //   fori(0,3*N){
    // 	real dn = abs(n[i]-np[i]);

    // 	if(dn>noiseMax) noiseMax = dn;
	
    // 	noiseSum += dn;
	
    // 	np[i] = n[i];
    //   }
    //   noiseSum = (noiseSum)/real(3*N);      
    //   if(noiseSum<1e-3) break;
    // }    
  }

  cerr<<"Convergence steps: "<<i<<endl;
  this->compNoise(1.0/invz2, N, i-1);
  
  
  //cudaFree(d_temp);
  return (real*)noise.d_m;
}

  
/*Computes the brownian noise in the current Lanczos iteration,
  z2 is the norm of the initial random vector,
  iter is the current iteration*/

void BrownianNoiseLanczos::compNoise(real z2, uint N, uint iter){
  real alpha = 1.0;
  real beta = 0.0;

  fori(0,iter) hdiag_temp[i] = hdiag[i];
  
  /**** y = ||z||_2 * Vm · H^1/2 · e_1 *****/


  /*Compute H^1/2 by diagonalization***********/
  /* H^1/2 = P · Hdiag^1/2 · P^T */
  /******WITH CUSOLVER SVD****************/
  fori(0,iter*iter) H.data[i] = 0;
  fori(0,iter){
    H.data[i*iter+i] = hdiag_temp[i];
    if((uint)i<iter-1){
      H.data[(i+1)*iter+i] = hsup[i];
    }
    if(i>0){
      H.data[(i-1)*iter+i] = hsup[i-1];
    }
  }

  H.upload(iter*iter);
  /*Diagonalize H*/
  cudaDeviceSynchronize();
  cusolverDngesvd(solver_handle, 'A', 'A', iter, iter,
		  H.d_m, iter,
		  hdiag_temp.d_m,
		  P.d_m, iter,
		  Pt.d_m, iter,
		  d_work, h_work_size, nullptr ,d_info);  
  cudaDeviceSynchronize();
  
  hdiag_temp.download(iter);
  /********************************IN CPU WITH CUSTOM METHOD***********************/
  //triDiagEig(hdiag_temp.data, hsup.data, P.data, iter);

  
  /*sqrt H*/
  fori(0, iter){
    if(hdiag_temp[i]>0)
      hdiag_temp[i] = sqrt(hdiag_temp[i]);
    else hdiag_temp[i] = 0.0;
  }    
  /* store in hdiag_temp -> H^1/2·e1 = P·Hdiag_Temp^1/2·Pt·e1*/
  fori(0,iter*iter) H.data[i] = 0;
  fori(0,iter) H.data[i*iter+i] = hdiag_temp[i];
  
  H.upload(iter*iter);
  
  cublasgemm(cublas_handle,
    	     CUBLAS_OP_N, CUBLAS_OP_N,
    	     iter,iter,iter,
    	     &alpha,
    	     H, iter,
    	     Pt, iter,
    	     &beta,
    	     Htemp,  iter);
  cublasgemm(cublas_handle,
    	     CUBLAS_OP_N,  CUBLAS_OP_N,
    	     iter, iter, iter,
    	     &alpha,
    	     P, iter,
    	     Htemp, iter,
    	     &beta,
   	     H, iter);
  /*Now H contains H^1/2*/  
  /*Store H^1/2 · e1 in hdiag_temp*/
  /*hdiag_temp = H^1/2·e1*/
  cudaMemcpy(hdiag_temp.d_m, H.d_m, iter*sizeof(real), cudaMemcpyDeviceToDevice);
  
  /*noise = ||z||_2 * Vm · H^1/2 · e1 = Vm · (z2·hdiag_temp)*/
  cublasStatus_t st = cublasgemv(cublas_handle, CUBLAS_OP_N,
				 3*N, iter,
				 &z2,
				 V, 3*N,
				 hdiag_temp.d_m, 1,
				 &beta,
				 (real*)noise.d_m, 1);
  
  // noise.download();
  // ofstream out("noiseLanc.dat");
  // fori(0,3*N)
  //   out<<((real*)noise.data)[i]<<endl;
  // exit(0);

}

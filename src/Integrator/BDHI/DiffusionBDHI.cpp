/*Raul P. Pelaez 2016. Diffusion Handler implementation, see .h*/

#include"DiffusionBDHI.h"

using namespace brownian_hy_euler_maruyama_ns;

/*********************************DIFFUSION HANDLER****************************/

/*Diffusion constructor*/
Diffusion::Diffusion(Matrixf D0in, uint N):
  N(N) /*Number of particles, not degrees of freedom*/
{
  this->D0 = D0in;
  if(!D0.isSym()){
    cerr<<"D0 Matrix must be symmetric!!"<<endl;
    exit(1);
  }
  if(D0.n != 9){
    cerr<<"D0 must be 3x3!!"<<endl;
    exit(1);
  }     

  
  params.D0 = D0[0][0];/*All D0 should be in constant memory, not just the first element*/
  cerr<<"\tD0: "<<D0[0][0]<<endl;
  params.rh = 0.5*gcnf.sigma;
  initRPYGPU(params);
}


DiffusionFullMatrix::DiffusionFullMatrix(Matrixf D0in, uint N):
  Diffusion(D0in,N)
{
  /*Store the matrix only in FULL mode*/
  D = Matrixf(3*N,3*N);

  D.fill_with(0.0f);
  D.upload();
}
DiffusionMatrixFree::DiffusionMatrixFree(Matrixf D0in, uint N):
  Diffusion(D0in,N)
{

}

void DiffusionFullMatrix::compute(){
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
   // exit(0);  
}


/*Computes Dv = D·v, it is not needed to pass handle in a matrix free method*/
void DiffusionFullMatrix::dot(real *v, real *Dv,cublasHandle_t handle, cudaStream_t st){
  if(st) cublasSetStream(handle, st);    
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


void DiffusionMatrixFree::dot(real *v, real *Dv, cublasHandle_t handle, cudaStream_t st){
  diffusionDotGPU(pos, (real3*)v, (real3*)Dv, N, st);
}


void Diffusion::divergence(real *res, real *noise, cublasHandle_t handle, cudaStream_t st){

  //diffusionDotGPU(pos, (real3*)noise, (real3*)res, N, st, true); //For RDF
  divergenceGPU(pos, (real3* )res, N);
}







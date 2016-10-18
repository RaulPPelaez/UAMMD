/*
Raul P. Pelaez 2016.
Force evaluator handler.
Takes a function to compute the force, a number of sampling points and a
 cutoff distance. Evaluates the force and saves it to upload as a texture 
 with interpolation.

TODO:
100- Use texture objects instead of creating and binding the texture in 
     initGPU.

 */
#include"Potential.h"
#include<fstream>
#include<cstring>

Potential::Potential(std::function<real(real)> Ffoo,
		     std::function<real(real)> Efoo,
		     int N, real rc):
  N(N), F(N), E(N),
  FGPU(nullptr), EGPU(nullptr),
  texForce(0),texEnergy(0),
  forceFun(Ffoo), energyFun(Efoo)
{
  F[0] = 0.0;
  E[0] = 0.0;

  real dr2 = rc*rc/(real)N;
  real r2 = 0.5*dr2;
  fori(1,N){
    r2 += dr2;
    F[i] =(float) forceFun(r2);
    E[i] =(float) energyFun(r2);
  }
  F[N-1] = 0.0;
  E[N-1] = 0.0;



  cudaChannelFormatDesc channelDesc;
  channelDesc = cudaCreateChannelDesc(32, 0,0,0, cudaChannelFormatKindFloat);

  
  gpuErrchk(cudaMallocArray(&FGPU,
			    &channelDesc,
			    N,1));
  gpuErrchk(cudaMallocArray(&EGPU,
			    &channelDesc,
			    N,1));
  
  gpuErrchk(cudaMemcpyToArray(FGPU, 0,0, F.data(), N*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToArray(EGPU, 0,0, E.data(), N*sizeof(float), cudaMemcpyHostToDevice));



  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;

    
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.normalizedCoords = 1;

  resDesc.res.array.array = FGPU;
  gpuErrchk(cudaCreateTextureObject(&texForce, &resDesc, &texDesc, NULL));
  resDesc.res.array.array = EGPU;
  gpuErrchk(cudaCreateTextureObject(&texEnergy, &resDesc, &texDesc, NULL));
    
  if(!texForce || !texEnergy ||
     !FGPU || !EGPU){
    cerr<<"Error in potential creation!!!"<<endl;
    exit(1);
  }
  
}

void Potential::print(){
  ofstream out("potential.dat");
  fori(0,N) out<<F[i]<<" "<<E[i]<<"\n";
  out.close();
}

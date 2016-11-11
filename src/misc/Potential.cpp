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
#include<algorithm>

Potential::Potential(std::function<real(real)> Ffoo,
		     std::function<real(real)> Efoo,
		     int N, real rc, uint ntypes):
  N(N), F(N), E(N),
  FGPU(nullptr), EGPU(nullptr),
  texForce(0),texEnergy(0),
  forceFun(Ffoo), energyFun(Efoo),
  ntypes(ntypes), potParams(ntypes*ntypes)
{
  F[0] = 0.0;
  E[0] = 0.0;

  real dr2 = rc*rc/(real)N;
  real r2 = 0.5*dr2;
  real sig2 = gcnf.sigma*gcnf.sigma;
  fori(1,N){
    r2 += dr2;
    F[i] =(float) forceFun(r2/sig2);
    E[i] =(float) energyFun(r2/sig2);
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
  
  gpuErrchk(cudaMemcpyToArray(FGPU, 0,0, F.data, N*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToArray(EGPU, 0,0, E.data, N*sizeof(float), cudaMemcpyHostToDevice));



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
  cudaCreateTextureObject(&texForce, &resDesc, &texDesc, NULL);
  resDesc.res.array.array = EGPU;
  cudaCreateTextureObject(&texEnergy, &resDesc, &texDesc, NULL);


  F.upload();
  E.upload();
  // if(!texForce || !texEnergy ||
  //    !FGPU || !EGPU){
  //   cerr<<"Error in potential creation!!!"<<endl;
  //   exit(1);
  // }

  potParams.fill_with(make_real2(1));

}

void Potential::print(){
  ofstream out("potential.dat");
  fori(0,N) out<<F[i]<<" "<<E[i]<<"\n";
  out.close();
}

void Potential::setPotParam(uint i, uint j, real2 params){
  if(i>=ntypes || j>=ntypes){
    cerr<<"WARNING: Cannot set particle type "<<i<<","<<j<<". Only "<<ntypes<<" particle types in the system"<<endl;
    return;
  }
  potParams[i+ntypes*j] = params;
  potParams[j+ntypes*i] = params;
  potParams.upload();
}

real2* Potential::getPotParams(){
  potParams.upload();
  return potParams.d_m;

}

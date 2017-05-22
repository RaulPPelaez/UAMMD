/*
Raul P. Pelaez 2016.Potentials 

 */
#include"Potential.cuh"
#include<fstream>
#include<cstring>
#include<algorithm>




TablePotential::TablePotential(std::function<real(real, real)> Ffoo,
			       std::function<real(real, real)> Efoo,
			       int N, real rc):
  N(N), F(N), E(N),
  FGPU(nullptr), EGPU(nullptr),
  texForce(0),texEnergy(0),
  forceFun(Ffoo), energyFun(Efoo),
  rc(rc)
{
  F[0] = 0.0f;
  E[0] = 0.0f;

  real dr2 = rc*rc/(real)N;
  real r2 = 0.5*dr2;
  real sig2 = gcnf.sigma*gcnf.sigma;
  fori(1,N){
    r2 += dr2;
    F[i] =(float) forceFun(r2/sig2, rc);
    E[i] =(float) energyFun(r2/sig2, rc);
  }
  F[N-1] = 0.0;
  E[N-1] = 0.0;


  cudaChannelFormatDesc channelDesc;
  channelDesc = cudaCreateChannelDesc(32, 0,0,0, cudaChannelFormatKindFloat);

  /*If the system doesnt support texture objects,
    the textures will be initialized as references when needed*/
  if(sysInfo.cuda_arch>210){
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
    gpuErrchk(cudaCreateTextureObject(&texForce, &resDesc, &texDesc, NULL));
    resDesc.res.array.array = EGPU;
    gpuErrchk(cudaCreateTextureObject(&texEnergy, &resDesc, &texDesc, NULL));
  }

  F.upload();
  E.upload();
  // if(!texForce || !texEnergy ||
  //    !FGPU || !EGPU){
  //   cerr<<"Error in potential creation!!!"<<endl;
  //   exit(1);
  // }
  

}

TablePotential::~TablePotential(){

   cudaFreeArray(FGPU);  
   cudaFreeArray(EGPU);
  
   cudaDestroyTextureObject(texForce);
   cudaDestroyTextureObject(texEnergy);
}
void TablePotential::print(){
  ofstream out("potential.dat");
  fori(0,N) out<<F[i]<<" "<<E[i]<<"\n";
  out.close();
}

//Force between two particles, depending on square distance between them
// this function is only called on construction, so it doesnt need to be optimized at all
//Distance is in units of sigma

real forceLJ(real r2, real rcut){
  real invr2 = 1/(r2);

  real invr6 = invr2*invr2*invr2;
  real invr8 = invr6*invr2;

  real invrc13 = pow(1/rcut, 13);
  real invrc7 = pow(1/rcut, 7);
  
  real fmod = -48.0*invr8*invr6 + 24.0*invr8;

  if(rcut>2.0){
    real invr = sqrt(invr2);
    real fmodcorr = 48.0*invr*invrc13 - 24.0*invr*invrc7;
  //(f(r)-f(rcut))/r
    return fmod+fmodcorr;
  }
  else
    return fmod;
}
real energyLJ(real r2, real rcut){
  real r = sqrt(r2);
  real invr2 = 1.0/r2;
  real invr6 = invr2*invr2*invr2;
  //real E =  2.0f*invr6*(invr6-1.0f);
  //potential as u(r)-(r-rcut)*f(rcut)-u(rcut) 
  real E = 2.0*(invr6*(invr6-1.0));
  real Ecorr = 2.0*(
		      -(r-rcut)*(-24.0*pow(1.0/rcut, 13)+12.0*pow(1.0/rcut, 7))
		      -pow(1.0/rcut, 6)*(pow(1.0/rcut, 6)-1.0));
  if(rcut>2.0)
    return E+Ecorr;
  else
    return E;
}

/*Raul P. Pelaez 2016. Two step velocity VerletNVT Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  Currently uses a BBK thermostat to maintain the temperature.
  Solves the following differential equation using a two step velocity verlet algorithm, see GPU code:
      X[t+dt] = X[t] + v[t]·dt
      v[t+dt]/dt = -gamma·v[t] - F(X) + sigma·G
  gamma is a damping constant, sigma = sqrt(2·gamma·T) and G are normal distributed random numbers with var 1.0 and mean 0.0.

TODO:
100-Implement thermostat from https://arxiv.org/pdf/1212.1244.pdf (change between them through an argument enum)
100- Allow velocities from outside
90-  Implement more thermostats

*/
#include "VerletNVT.cuh"
#include<cub/iterator/transform_input_iterator.cuh>
#include<cub/device/device_reduce.cuh>
#include"utils/GPUutils.cuh"
#include"globals/globals.h"
#include"utils/vector_overloads.h"
#include"utils/helper_gpu.cuh"

#ifndef SINGLE_PRECISION
#define curandGenerateNormal curandGenerateNormalDouble
#endif


VerletNVT::VerletNVT():VerletNVT(gcnf.N, gcnf.L, gcnf.dt, gcnf.gamma){}


VerletNVT::VerletNVT(int N, real3 L, real dt, real gamma):
  /*After initializing the base class, you have access to things like N, L...*/
  Integrator(N, L, dt, 128), 
  noise(N +((3*N)%2)),
  gamma(gamma),
  T(gcnf.T),
  d_temp_storage(nullptr),
  temp_storage_bytes(0),
  d_K(nullptr)
{
  cerr<<"Initializing Verlet NVT Integrator..."<<endl;

  cerr<<"\tSet T="<<gcnf.T<<endl;

  this->noiseAmp = sqrt(dt*0.5)*sqrt(2.0*gamma*T);
  
  /*Init rng*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, grng.next());
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), 0.0, 1.0);
  if(vel.size()!=N){
    vel = Vector3(N);
    noise.download();
    /*Distribute the velocities according to the temperature*/
    double vamp = sqrt(3.0*T);
    /*Create velocities*/
    vel.fill_with(make_real3(0.0));
    fori(0,N){
      vel[i] = vamp*noise[i];
      if(gcnf.D2) vel[i].z = 0;
    }
    vel.upload();
    curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), 0.0, 1.0);
  }
  cerr<<"Verlet NVT Integrator\t\tDONE!!\n"<<endl;
}

VerletNVT::~VerletNVT(){
  curandDestroyGenerator(rng);
  cudaFree(d_temp_storage);
  cudaFree(d_K);
}


namespace VerletNVT_ns{

  /*Integrate the movement*/
  template<int step>
  __global__ void integrateGPU(real4 __restrict__  *pos,
			       real3 __restrict__ *vel,
			       const real4 __restrict__  *force,
			       const real3 __restrict__ *noise, int N,
			       real dt, real gamma, bool D2){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    /*Half step velocity*/
    vel[i] += (make_real3(force[i])-gamma*vel[i])*dt*real(0.5) + noise[i];

    if(D2) vel[i].z = real(0.0);
    /*In the first step, upload positions*/
    if(step==1)
      pos[i] += make_real4(vel[i])*dt;
        
  }


};

//The integration process can have two steps
void VerletNVT::update(){
  if(steps==0)
    for(auto forceComp: interactors) forceComp->sumForce();
  
  steps++;
  uint nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0);

  /**First integration step**/
  /*Gen noise*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), noiseAmp);
  
  VerletNVT_ns::integrateGPU<1><<<nblocks, nthreads>>>(pos, vel, force, noise, N, dt, gamma, gcnf.D2);
  /**Compute all the forces**/
  /*Reset forces*/
  cudaMemset((void *)force.d_m, 0, N*sizeof(real4));
  for(auto forceComp: interactors) forceComp->sumForce();

  /**Second integration step**/
  /*Gen noise*/
  curandGenerateNormal(rng, (real*) noise.d_m, 3*N + ((3*N)%2), real(0.0), noiseAmp);
  VerletNVT_ns::integrateGPU<2><<<nblocks, nthreads>>>(pos, vel, force, noise, N,  dt, gamma, gcnf.D2);
}


namespace VerletNVT_ns{

  /*Returns the squared of each element in a real3*/
  struct dot_functor{
    inline __device__ real3 operator() (const real3 &a) const{
      return a*a;
    }

  };

};

real VerletNVT::sumEnergy(){  
  VerletNVT_ns::dot_functor dot_op;
  cub::TransformInputIterator<real3, VerletNVT_ns::dot_functor, real3*> dot_iter(vel.d_m, dot_op);
  if(!d_temp_storage){    
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dot_iter, d_K, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);    
    if(!d_K)
      cudaMalloc(&d_K, sizeof(real3));
  }
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dot_iter, d_K, N);
  real3 K;
  cudaMemcpy(&K, d_K, sizeof(real3), cudaMemcpyDeviceToHost);

  return 0.5f*(K.x+K.y+K.z)/(real)N;
}


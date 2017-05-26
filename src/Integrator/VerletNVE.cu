/*Raul P. Pelaez 2016. Two step velocity VerletNVE Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation:
      X[t+dt] = X[t] +v[t]·dt+0.5·a[t]·dt^2
      v[t+dt] = v[t] +0.5·(a[t]+a[t+dt])·dt
TODO:
100- Allow velocities from outside
100- Allow to set the initial velocity instead of the energy
*/
#include "VerletNVE.cuh"
#include<cub/iterator/transform_input_iterator.cuh>
#include<cub/device/device_reduce.cuh>
#include"utils/GPUutils.cuh"
#include"globals/globals.h"
#include"utils/vector_overloads.h"
#include"utils/helper_gpu.cuh"

/*Constructor, Dont forget to initialize the base class Integrator!*/
/*You can use anything in gcnf at any time*/
VerletNVE::VerletNVE():
  VerletNVE(gcnf.N, gcnf.L, gcnf.dt){}

VerletNVE::VerletNVE(int N, real3 L, real dt):
  Integrator(N, L, dt, 128), E(gcnf.E),
  d_temp_storage(nullptr),
  temp_storage_bytes(0),
  d_K(nullptr)
{
  cerr<<"Initializing Verlet NVE Integrator..."<<endl;

  if(vel.size()!=N){
    /*Create the velocity if you need it*/
    vel = Vector3(N);  vel.fill_with(make_real3(0.0));  vel.upload();
  }
    
  cerr<<"\tSet E="<<E<<endl;
  cerr<<"Verlet NVE Integrator\t\tDONE!!\n"<<endl;
}

VerletNVE::~VerletNVE(){
  cudaFree(d_K);
  cudaFree(d_temp_storage);
  cerr<<"Destroying VerletNVE...";
  cerr<<"\tDONE!!"<<endl;
}



namespace VerletNVE_ns{
  template<int step>
  __global__ void integrateGPU(real4 __restrict__  *pos,
				real3 __restrict__ *vel,
				const real4 __restrict__  *force,
				int N, real dt, bool D2){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    /*Half step velocity*/
    vel[i] += make_real3(force[i])*dt*real(0.5);
    
    if(D2) vel[i].z = real(0.0); //2D
    
    /*In the first step, upload positions*/
    if(step==1)
      pos[i] += make_real4(vel[i])*dt;
  }


};
//The integration process can have two steps
void VerletNVE::update(){
  /*Set the energy in the first step*/
  if(steps==0){
    /*In the first step, compute the force and energy in the system
      in order to adapt the initial kinetic energy to match the input total energy
      E = U+K */
    real U = 0.0;
    for(auto forceComp: interactors) U += forceComp->sumEnergy();
    real K = abs(E-U);
    /*Distribute the velocities accordingly*/
    real vamp = sqrt(2.0*K/3.0);
    /*Create velocities*/
    vel.fill_with(make_real3(real(0.0)));
    fori(0,N){
      vel[i].x = vamp*grng.gaussian(0.0, 1.0);
      vel[i].y = vamp*grng.gaussian(0.0, 1.0);
      vel[i].z = vamp*grng.gaussian(0.0, 1.0);
    }
    vel.upload();
    cudaMemset(force.d_m, 0, N*sizeof(real4));
  }
  
  steps++;
  uint nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  uint nblocks = N/nthreads +  ((N%nthreads!=0)?1:0);

  
  /**First integration step**/
  VerletNVE_ns::integrateGPU<1><<<nblocks, nthreads>>>(pos, vel, force, N, dt, gcnf.D2);
  /**Reset the force**/
  /*The integrator is in charge of resetting the force when it needs, an interactor always sums to the current force*/
  cudaMemset(force.d_m, 0, N*sizeof(real4));
  /**Compute all the forces**/
  for(auto forceComp: interactors) forceComp->sumForce();
  
  /**Second integration step**/
  VerletNVE_ns::integrateGPU<2><<<nblocks, nthreads>>>(pos, vel, force, N, dt, gcnf.D2);
}





namespace VerletNVE_ns{
  /*Returns the squared of each element in a real3*/
  struct dot_functor{
    inline __device__ real3 operator() (const real3 &a) const{
      return a*a;
    }
  };
};

real VerletNVE::sumEnergy(){  
  VerletNVE_ns::dot_functor dot_op;
  cub::TransformInputIterator<real3, VerletNVE_ns::dot_functor, real3*> dot_iter(vel.d_m, dot_op);
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

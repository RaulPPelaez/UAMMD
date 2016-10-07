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
100- Allow velocities from outside
90-  Implement more thermostats
80-  Gamma should be chosen by the user
70- Improve the intial conditions for the velocity, should be normal, not uniform.
*/
#include "VerletNVT.h"

/*Each module should have its own namespace*/
using namespace verlet_nvt_ns;

VerletNVT::VerletNVT():
  Integrator(), /*After initializing the base class, you have access to things like N, L...*/
  noise(N +((3*N)%2))
{
  cerr<<"Initializing Verlet NVT Integrator..."<<endl;


  cerr<<"\tSet T="<<gcnf.T<<endl;

  gamma = gcnf.gamma;

  /*Set params and init GPU parameters*/
  params.gamma = gamma;
  params.dt = dt;
  params.T = gcnf.T;
  params.noiseAmp = sqrt(dt*0.5f)*sqrt(2.0f*gamma*params.T);
  params.N = N;
  params.L = L;
  
  /*Init rng*/
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, grng.next());
  /*This shit is obscure, curand will only work with an even number of elements*/
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  if(vel.size()!=N){
    vel = Vector3(N);
    noise.download();
    /*Distribute the velocities according to the temperature*/
    float vamp = sqrt(3.0f*params.T);
    /*Create velocities*/
    vel.fill_with(make_float3(0.0f));
    fori(0,N){
      vel[i] = vamp*noise[i];
    }
    vel.upload();
    curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  }
  
  initGPU(params);
  cerr<<"Verlet NVT Integrator\t\tDONE!!\n"<<endl;
}

VerletNVT::~VerletNVT(){
  curandDestroyGenerator(rng);
}

//The integration process can have two steps
void VerletNVT::update(){
  if(steps==0)
    for(auto forceComp: interactors) forceComp->sumForce();
  
  steps++;
  if(steps%1000==0) cerr<<"\rComputing step: "<<steps<<"   ";

  /**First integration step**/
  /*Gen noise*/
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  integrateGPU(pos, vel, force, noise, N, 1);
  /**Compute all the forces**/
  
  /*Reset forces*/
  cudaMemset((void *)force.d_m, 0, N*sizeof(float4));
  for(auto forceComp: interactors) forceComp->sumForce();
  
  /**Second integration step**/
  /*Gen noise*/
  curandGenerateNormal(rng, (float*) noise.d_m, 3*N + ((3*N)%2), 0.0f, 1.0f);
  integrateGPU(pos, vel, force, noise, N, 2);
  

  
 }


float VerletNVT::sumEnergy(){
  return computeKineticEnergyGPU(vel, N);
}


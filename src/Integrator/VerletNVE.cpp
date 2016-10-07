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
#include "VerletNVE.h"

using namespace verlet_nve_ns;

/*Constructor, Dont forget to initialize the base class Integrator!*/
/*You can use anything in gcnf at any time*/
VerletNVE::VerletNVE():
  Integrator(), E(gcnf.E){
  cerr<<"Initializing Verlet NVE Integrator..."<<endl;

  if(vel.size()!=N){
    /*Create the velocity if you need it*/
    vel = Vector3(N);  vel.fill_with(make_float3(0.0f));  vel.upload();
  }
  cerr<<"\tSet E="<<E<<endl;

  /*All of this come from global config, and are defined as members in Integrator*/
  params.dt = dt;
  params.L = L;
  params.N = N;
  initGPU(params);
  cerr<<"Verlet NVE Integrator\t\tDONE!!\n"<<endl;
}

VerletNVE::~VerletNVE(){
  cerr<<"Destroying VerletNVE...";
  cerr<<"\tDONE!!"<<endl;

}

//The integration process can have two steps
void VerletNVE::update(){
  /*Set the energy in the first step*/
  if(steps==0){
    /*In the first step, compute the force and energy in the system
      in order to adapt the initial kinetic energy to match the input total energy
      E = U+K */
    float U = 0.0f;
    for(auto forceComp: interactors) U += forceComp->sumEnergy();
    
    float K = abs(E-U);
    /*Distribute the velocities accordingly*/
    float vamp = sqrt(2.0f*K);
    /*Create velocities*/
    vel.fill_with(make_float3(0.0f));
    fori(0,N){
      vel[i].x = vamp*(grng.uniform(-1.0f, 1.0f));
      vel[i].y = vamp*(grng.uniform(-1.0f, 1.0f));
      vel[i].z = vamp*(grng.uniform(-1.0f, 1.0f));
    }
    vel.upload();
  }
  
  steps++;
  if(steps%1000==0) cerr<<"\rComputing step: "<<steps<<"   ";
  /**First integration step**/
  integrateGPU(pos, vel, force, N, 1);
  /**Reset the force**/
  /*The integrator is in charge of resetting the force when it needs, an interactor always sums to the current force*/
  cudaMemset(force.d_m, 0.0f, N*sizeof(float4));

  /**Compute all the forces**/
  for(auto forceComp: interactors) forceComp->sumForce();
  /**Second integration step**/
  integrateGPU(pos, vel, force, N, 2);
}


float VerletNVE::sumEnergy(){
  /*The only apportation to the energy is kinetic*/
  return computeKineticEnergyGPU(vel, N);
}

/*You can hijack the writing to disk process like this and perform a custom write,
  maybe to add something or to completly change the process*/
void VerletNVE::write(bool block){
  Integrator::write(block);
   // vel.download();
   // float3 res = make_float3(0.0f);
   // fori(0,N){
   //   res += vel[i];
   // }

   // cout<<(res.x/(float)N)<<" "<<res.y/(float)N<<" "<<res.z/(float)N<<endl;
}

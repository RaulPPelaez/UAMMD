/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation:
      X[t+dt] = X[t] +v[t]·dt+0.5·a[t]·dt^2
      v[t+dt] = v[t] +0.5·(a[t]+a[t+dt])·dt

*/
#include "TwoStepVelVerlet.h"


TwoStepVelVerlet::TwoStepVelVerlet(shared_ptr<Vector<float4>> pos,
				   shared_ptr<Vector<float4>> force, uint N, float L, float dt, float T):
  vel(N),
  Integrator(pos, force, N, L, dt){

  float vamp = sqrt(3.0f*T);
  /*Create velocities*/
  vel.fill_with(make_float3(0.0f));
   fori(0,N){ 
     vel[i].x = vamp*(2.0f*(rand()/(float)RAND_MAX)-1.0f);
     vel[i].y = vamp*(2.0f*(rand()/(float)RAND_MAX)-1.0f);
     vel[i].z = vamp*(2.0f*(rand()/(float)RAND_MAX)-1.0f);
   }
  vel.upload();
  for(auto forceComp: interactors) forceComp->sumForce();
}
TwoStepVelVerlet::~TwoStepVelVerlet(){}

//The integration process can have two steps
void TwoStepVelVerlet::update(){
  
  steps++;
  if(steps%1000==0) cerr<<"\rComputing step: "<<steps<<"   ";
  /**First integration step**/
  integrateTwoStepVelVerletGPU(pos->d_m, vel, force->d_m, dt, N, 1);

  /**Compute all the forces**/
  for(auto forceComp: interactors) forceComp->sumForce();

  /**Second integration step**/
  integrateTwoStepVelVerletGPU(pos->d_m, vel, force->d_m, dt, N, 2);//, steps%10 ==0 && steps<10000 && steps>1000);

  // if(steps%1000==0){
  //   vel.download();
  //   float v2t = 0.0f;
  //   fori(0,N){
  //     v2t += dot(vel[i],vel[i]);
  //   }
  //   cerr<<v2t/(3.0f*N)<<endl;

  // }
}


float TwoStepVelVerlet::sumEnergy(){
  return computeKineticEnergyTwoStepVelVerlet(vel, N);
}

void TwoStepVelVerlet::write(bool block){
  Integrator::write(block);

  // static ofstream out("mom.dat");
  // static ofstream out2("vel.dat");  
  // vel.download();
  // float3 Ptot = make_float3(0.0f);
  // fori(0,N){
  //   Ptot += vel[i];
  //   out2<<vel[i].x<<" "<<vel[i].y<<" "<<vel[i].z<<"\n";
  // }
  // // out<<endl;
  // Ptot/=N;
  // out<<Ptot.x<<" "<<Ptot.y<<" "<<Ptot.z<<endl;
}

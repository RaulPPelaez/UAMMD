/*Raul P. Pelaez 2016. TwoStepVelVerlet Integrator derived class

  Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of creating the velocities and keep the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
  
  TODO:
   Maybe the velocities should be outside the module, handled as the positions.

 */
#include "TwoStepVelVerlet.h"


TwoStepVelVerlet::TwoStepVelVerlet(shared_ptr<Vector<float4>> pos,
				     shared_ptr<Vector<float4>> force, uint N, float L, float dt):
  vel(N),
  Integrator(pos, force, N, L, dt){

  /*Create velocities*/
  vel.fill_with(make_float3(0.0f));
  fori(0,N){ 
    vel[i].x = 0.1f*(2.0f*(rand()/(float)RAND_MAX)-1.0f);
    vel[i].y = 0.1f*(2.0f*(rand()/(float)RAND_MAX)-1.0f);
    vel[i].z = 0.1f*(2.0f*(rand()/(float)RAND_MAX)-1.0f);
  }
  vel.upload();
}
TwoStepVelVerlet::~TwoStepVelVerlet(){}

//The integration process can have two steps
void TwoStepVelVerlet::update(){
  
  steps++;
  if(steps%500==0) cerr<<"\rComputing step: "<<steps<<"   ";
  integrateTwoStepVelVerletGPU(pos->d_m, vel, force->d_m, dt, N, 1);

  for(auto forceComp: interactors) forceComp->sumForce();

  integrateTwoStepVelVerletGPU(pos->d_m, vel, force->d_m, dt, N, 2);//, steps%10 ==0 && steps<10000 && steps>1000);


}
// void TwoStepVelVerlet::updateSecondStep(){
//   integrateTwoStepVelVerletGPU(pos->d_m, vel, force->d_m, dt, N, 2);//, steps%10 ==0 && steps<10000 && steps>1000);
// }



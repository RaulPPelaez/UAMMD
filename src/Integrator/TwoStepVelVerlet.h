/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator derived class

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation:
      X[t+dt] = X[t] +v[t]·dt+0.5·a[t]·dt^2
      v[t+dt] = v[t] +0.5·(a[t]+a[t+dt])·dt

TODO:
100- The initial velocities should be related with a temperature
*/
#ifndef TWOSTEPVELVERLETINTEGRATOR_H
#define TWOSTEPVELVERLETINTEGRATOR_H
#include "utils/utils.h"
#include "Integrator.h"
#include "TwoStepVelVerletGPU.cuh"
#include<thread>

class TwoStepVelVerlet: public Integrator{
public:
  TwoStepVelVerlet(shared_ptr<Vector<float4>> pos,
		   shared_ptr<Vector<float4>> force, uint N, float L, float dt);
  ~TwoStepVelVerlet();

  void update() override;

private:
  //The only additional information to store is the velocity
  Vector<float3> vel;
};



#endif

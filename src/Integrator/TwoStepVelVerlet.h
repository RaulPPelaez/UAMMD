/*Raul P. Pelaez 2016. Two step velocity Verlet Integrator derived class

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation:
      X[t+dt] = X[t] +v[t]·dt+0.5·a[t]·dt^2
      v[t+dt] = v[t] +0.5·(a[t]+a[t+dt])·dt
*/
#ifndef TWOSTEPVELVERLETINTEGRATOR_H
#define TWOSTEPVELVERLETINTEGRATOR_H
#include "utils/utils.h"
#include "Integrator.h"
#include "TwoStepVelVerletGPU.cuh"
#include<thread>

class TwoStepVelVerlet: public Integrator{
public:
  TwoStepVelVerlet(Vector4Ptr pos,
		   Vector4Ptr force, uint N, float L, float dt, float T);
  ~TwoStepVelVerlet();

  void update() override;
  //Returns the kinetic energy
  float sumEnergy() override;

  void write(bool block);
private:
  //The only additional information to store is the velocity
  Vector3 vel;
};



#endif

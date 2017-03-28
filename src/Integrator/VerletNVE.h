/*Raul P. Pelaez 2016. Two step velocity VerletNVE Integrator derived class

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  
  Solves the following differential equation using a two step velocity verlet algorithm:
      X[t+dt] = X[t] +v[t]·dt+0.5·a[t]·dt^2
      v[t+dt] = v[t] +0.5·(a[t]+a[t+dt])·dt
*/
#ifndef VERLETNVE_H
#define VERLETNVE_H
#include "globals/defines.h"
#include "utils/utils.h"
#include "Integrator.h"
#include "VerletNVEGPU.cuh"


class VerletNVE: public Integrator{
public:
  VerletNVE();
  ~VerletNVE();

  void update() override;
  //Returns the kinetic energy
  real sumEnergy() override;
private:
  real E;
  verlet_nve_ns::Params params;
};



#endif

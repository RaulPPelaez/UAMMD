/*Raul P. Pelaez 2016. Two step velocity VerletNVT Integrator derived class

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of keeping the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
 
  This is similar to NVE but with a thermostat that keeps T constant by changing E.

  Currently uses a BBK thermostat to maintain the temperature.
  Solves the following differential equation using a two step velocity verlet algorithm, see GPU code:
      X[t+dt] = X[t] + v[t]·dt
      v[t+dt]/dt = -gamma·v[t] - F(X) + sigma·G
  gamma is a damping constant, sigma = sqrt(2·gamma·T) and G are normal distributed random numbers with var 1.0 and mean 0.0.

TODO:
100- Implement more thermostats, allow for selection of a thermostat on creation

*/
#ifndef VERLETNVT_H
#define VERLETNVT_H
#include "utils/utils.h"
#include "Integrator.h"
#include "VerletNVTGPU.cuh"
#include<curand.h>

class VerletNVT: public Integrator{
public:
  VerletNVT();
  ~VerletNVT();

  void update() override;
  //Returns the kinetic energy
  float sumEnergy() override;

  void setTemp(float Tnew){
    params.T = Tnew;
    params.noiseAmp = sqrt(dt*0.5f)*sqrt(2.0f*gamma*Tnew);
    initVerletNVTGPU(params);
  }
private:
  Vector3 vel, noise;
  
  float gamma; //Gamma is not stored in gcnf
  
  VNVTparams params;
  curandGenerator_t rng;
};



#endif

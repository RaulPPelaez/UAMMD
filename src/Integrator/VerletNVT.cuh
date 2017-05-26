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
#ifndef VERLETNVT_CUH
#define VERLETNVT_CUH
#include"globals/defines.h"
#include "utils/utils.h"
#include "Integrator.h"
#include<curand.h>

class VerletNVT: public Integrator{
public:
  VerletNVT();
  VerletNVT(int N, real3 L, real dt, real gamma);
  ~VerletNVT();

  void update() override;
  //Returns the kinetic energy
  real sumEnergy() override;

  void setTemp(real Tnew){
    this->T = Tnew;
    this->noiseAmp = sqrt(dt*0.5)*sqrt(2.0*gamma*Tnew);
  }
private:
  Vector3 noise; //Noise in single precision always  
  real gamma; //Gamma is not stored in gcnf
  real noiseAmp;
  real T;
  curandGenerator_t rng;


  /*For energy*/
  void *d_temp_storage;
  size_t temp_storage_bytes;
  real3 *d_K;
  
};

#endif

/*
  Raul P. Pelaez 2016. NBody Force Interactor

  Computes the interaction between all pairs in the system. Currently only gravitational force

TODO:
100- Allow custom force

*/

#ifndef NBODYFORCES_H
#define NBODYFORCES_H

#include"utils/utils.h"
#include"Interactor.h"
#include"NBodyForcesGPU.cuh"
#include"misc/Potential.h"

#include<cstdint>
#include<memory>
#include<functional>
#include<vector>

class NBodyForces: public Interactor{
public:
  NBodyForces(uint N, float L,
	       Vector4Ptr pos,
	       Vector4Ptr force);
  
  ~NBodyForces();

  void sumForce() override;
  float sumEnergy() override;
  float sumVirial() override;
private:
  void init();
  
  //  NBodyForcesParams params;
};
#endif

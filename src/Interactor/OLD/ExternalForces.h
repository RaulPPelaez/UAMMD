/*
  Raul P. Pelaez 2016. External forces applied to each particle, independent of other particles.

  Interactor is intended to be a module that computes and sums the forces acting on each particle
  due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

TODO:
  100- Allow for a custom force function from outside
*/

#ifndef EXTERNALFORCES_H
#define EXTERNALFORCES_H

#include"globals/defines.h"
#include"utils/utils.h"
#include"globals/globals.h"
#include"Interactor.h"
#include"ExternalForcesGPU.cuh"

#include<cstdint>
#include<memory>
#include<vector>

class ExternalForces: public Interactor{
public:
  ExternalForces();

  ~ExternalForces();

  void sumForce() override;
  real sumEnergy() override;
  real sumVirial() override;
private:
  void init();
  external_forces_ns::Params params;
};
#endif

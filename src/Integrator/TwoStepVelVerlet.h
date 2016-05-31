/*Raul P. Pelaez 2016. Integrator class

  Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of creating the velocities and keep the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
  
  TODO:
   Maybe the velocities should be outside the module, handled as the positions.

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
  //void updateSecondStep() override;

private:
  Vector<float3> vel;
};



#endif

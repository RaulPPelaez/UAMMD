/*Raul P. Pelaez 2016. Integrator class

  Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of creating the velocities and keep the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
  
  TODO:
   Maybe the velocities should be outside the module, handled as the positions.

*/
#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include "utils/utils.h"
#include "IntegratorGPU.cuh"
#include<thread>

void write_concurrent(float4 *pos, float L, uint N);
class Integrator{

public:
  Integrator(Vector<float4> *pos, Vector<float4> *force, uint N, float L,float dt);
  ~Integrator();

  void updateFirstStep();
  void updateSecondStep();

  void write(bool block = false);

private:
  //Pos and force are handled outside
  Vector<float4> *pos, *force;
  Vector<float3> vel;
  uint steps;
  uint N;
  float dt, L;

  std::thread *writeThread;
};



#endif

/*Raul P. Pelaez 2016. Integrator class

  Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of creating the velocities and keep the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk

 Currently Implemented integrators:
   1. Velocity Verlet
   2. Brownian Dynamics Euler Maruyama

  TODO:
   60- Maybe the velocities should be outside the module, handled as the positions.    
   100- Implement new integrators
*/
#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include "utils/utils.h"
#include "Interactor/Interactor.h"
#include<thread>
#include<memory>

void write_concurrent(float4 *pos, float L, uint N);
class Integrator{
public:
  Integrator();
  ~Integrator();

  virtual void update() = 0;
  //  virtual void updateSecondStep() = 0;

  void write(bool block = false);
  
  void addInteractor(shared_ptr<Interactor> an_interactor){
    interactors.push_back(an_interactor);
  }
protected:
  Integrator(shared_ptr<Vector<float4>> pos,
	     shared_ptr<Vector<float4>> force, uint N, float L, float dt);
  //Pos and force are handled outside
  shared_ptr<Vector<float4>> pos, force;
  vector<shared_ptr<Interactor>> interactors;
  uint steps;
  uint N;
  float dt, L;
  std::thread *writeThread;
};



#endif

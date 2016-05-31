/*Raul P. Pelaez 2016. Integrator abstract class

  Integrator is intended to be a separated module that handles the update of positions and applies forces on each particle via Interactors

  It takes care of keeping the positions updated.

  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk

  This is an abstract class that should be derived to implement new integrators, Integrator itself cannot be instanciated.

 Currently Implemented integrators:
   1. Velocity Verlet
   2. Brownian Dynamics Euler Maruyama (WIP)
   2. Brownian Dynamics Euler Maruyama with Hydrodynamics (WIP)

  TODO:
    90- Implement new integrators
    80- Create measurables to keep track of things, like energy or gdr
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
  
  //This function forwards the simulation one dt in time, must be overrided in each new implementation!
  virtual void update() = 0;

  void write(bool block = false);
  
  //The interactors can be called at any time from the integrator to compute the forces when needed.
  void addInteractor(shared_ptr<Interactor> an_interactor){
    interactors.push_back(an_interactor);
  }
protected:
  //Constructor to be called in the initialization list of the derived class
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

/*Raul P. Pelaez 2016. Integrator abstract class

  Integrator is intended to be a separated module that handles the update of positions and applies forces on each particle via Interactors

  It takes care of keeping the positions updated.

  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk

  This is an abstract class that should be derived to implement new integrators, Integrator itself cannot be instanciated.

 Currently Implemented integrators:
   1. Velocity Verlet NVE
   2. Velocity Verlet NVT with BBK thermostat
   3. Brownian Dynamics Euler Maruyama
   4. Brownian Dynamics Euler Maruyama with Hydrodynamics

  TODO:
    90- Implement new integrators
    80- Make a write engine that allows a custom format, like "XYZ", "%.3f\t%.3f\t%.3f"
*/


#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include"globals/defines.h"
#include "utils/utils.h"
#include "globals/globals.h"

#include "Interactor/Interactor.h"
#include<memory>

class Integrator{
public:
  //Constructor to be called in the initialization list of the derived class
  Integrator():
    Integrator(gcnf.N, gcnf.L, gcnf.dt){}
  Integrator(int N, real3 L, real dt, int BLOCKSIZE = 128):
    N(N), dt(dt), L(L), BLOCKSIZE(BLOCKSIZE){
    steps = 0;
  }

  ~Integrator(){}
  
  //This function forwards the simulation one dt in time, must be overrided in each new implementation!
  virtual void update() = 0;

  //this function returns any contribution to the system energy the integrator provides, such as the kinecit energy in MD
  virtual real sumEnergy() = 0;
  
  //The interactors can be called at any time from the integrator to compute the forces when needed.
  void addInteractor(shared_ptr<Interactor> an_interactor){
    std::cerr<<"Adding Interactor..."<<std::endl;
    interactors.push_back(an_interactor);
    std::cerr<<"\t";
    an_interactor->print();
    std::cerr<<std::endl;
  }
  vector<shared_ptr<Interactor>> getInteractors(){
    return interactors;
  }
protected:
  //Pos and force are handled outside
  vector<shared_ptr<Interactor>> interactors;
  uint steps;
  uint N;
  real dt;
  real3 L;
  string name;
  int BLOCKSIZE;
};



#endif

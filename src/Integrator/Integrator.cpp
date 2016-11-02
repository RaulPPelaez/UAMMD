/*Raul P. Pelaez 2016. Integrator class base implementation

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
*/

#include"Integrator.h"

Integrator::~Integrator(){
}

//Constructor to be called in the initialization list of the derived class
Integrator::Integrator():
  N(gcnf.N), dt(gcnf.dt), L(gcnf.L){
  steps = 0;
}


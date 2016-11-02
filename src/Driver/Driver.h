/*
Raul P. Pelaez 2016. MD simulator using Interactor and Integrator, example of usage.

NOTES:
The idea is to mix implementations of Integrator and Interactor to construct a simulation. 
For example create a TwoStepVelVerlet integrator and add a PairForces interactor with LJ to create a lennard jonnes gas MD simulation.


Once initialized this classes will perform a single task very fast as black boxes:

Integrator uploads the positions according to the velocities, forces and current positions.
Interactor computes the pair forces using the current positions according to the selected potential

The idea is for Integrator to control the positions and velocities and for Interactor to control the forces. Communicating each variable when needed. So if you need the vel. in the force computing you can pass it to your implementation of Interactor and use it in the force function accordingly.

The float4 forces contains fx, fy, fz, E. 
The float4 pos contains x,y,z,type

Several interactors can be added to an integrator, for example one interactor for pair forces, another for bonded forces..


TODO:
100- The force array should not be handled by the user
*/
#ifndef DRIVER_H
#define DRIVER_H

#include"globals/globals.h"
#include"globals/defines.h"
#include"Interactor/Interactor.h"
#include"Interactor/PairForces.h"
#include"Interactor/PairForcesDPD.h"
#include"Integrator/Integrator.h"
#include"Integrator/VerletNVE.h"
#include"Integrator/VerletNVT.h"
#include"Integrator/BrownianEulerMaruyama.h"
#include"Integrator/BrownianHydrodynamicsEulerMaruyama.h"
#include "Interactor/BondedForces.h"
#include "Interactor/NBodyForces.h"
#include "Interactor/ExternalForces.h"

#include"Measurable/Measurable.h"
#include"Measurable/EnergyMeasure.h"
#include"utils/utils.h"
#include<memory>
#include<thread>
#ifdef EXPERIMENTAL
#include"Interactor/Experimental/PairForcesAlt.h"
#endif



//This class handles writing to disk
class Writer{
public:
  Writer(){}
  ~Writer();
  //Write the current positions to disk, concurrently if block is false or not given
  void write(bool block = false);


  
private:
  void write_concurrent();
  std::thread writeThread;


};

//This class controls the flow of the simulation
class Driver{
protected:
  //Interactors are added to an integrator
  shared_ptr<Integrator> integrator;
  MeasurableArray measurables;
  uint step;

  //Call this once you have
  //set all the parameters needed by the simulation.
  //It initialices some arrays and does some work
  //that can only be done after the parameters are known.
  //Any change of parameters after a call to this function will produce
  //undefined behavior.
  void setParameters(); 
  
public:
  Writer writer;
  //The constructor configures and initializes the simulation
  Driver();
  ~Driver();
  //Move nsteps*dt forward in time
  void run(uint nsteps, bool relax = false);
  //Read an initial configuration from fileName, TODO
  void read(const char *fileName);
};


#endif

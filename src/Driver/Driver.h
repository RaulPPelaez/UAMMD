/*
Raul P. Pelaez 2016. Driver Holds all the simulation modules and is in charge of calling the integrator update method and writing to disk.

NOTES:
The idea is to mix implementations of Integrator and Interactor to construct a simulation. 
For example create a VerletNVT integrator and add a PairForces interactor with LJ to create a lennard jonnes liquid MD simulation.

Once initialized this modules will perform a single task very fast as black boxes:

Integrator uploads the positions according to the velocities, forces, current positions, etc.
Interactor computes forces according to the current simulation state (mainly the particle positions)

The float4 forces contains fx, fy, fz, non-used. 
The float4 pos contains x,y,z,type

Several interactors can be added to an integrator, for example one interactor for pair forces, another for bonded forces..
*/
#ifndef DRIVER_H
#define DRIVER_H

/*Base modules*/
#include"globals/globals.h"
#include"globals/defines.h"
#include"Interactor/Interactor.h"
#include"Integrator/Integrator.h"

/*A list of all the currently implemented modules*/
#include"Modules.h"

#include"utils/utils.h"
#include<memory>
#include<thread>



//This class handles writing to disk concurrently
class Writer{
public:
  Writer(){}
  ~Writer();
  //Write the current positions to disk, concurrently if block is false or not given
  void write(bool block = false);

  void synchronize();
  
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
  void identifyColors();
public:
  Writer writer;
  Driver();
  ~Driver();
  //Move nsteps*dt forward in time
  void run(uint nsteps, bool relax = false);
};


#endif

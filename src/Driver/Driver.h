/*
Raul P. Pelaez 2016. MD simulator using Interactor and Integrator, example of usage.


NOTES:
The idea is to use either Integrator or Interactor in another project as a module.

Once initialized this classes will perform a single task very fast as black boxes:

Integrator uploads the positions according to the velocities, forces and current positions.
Interactor computes the pair forces using the current positions according to the selected potential

The idea is for Integrator to control the positions and velocities and for Interactor to control the forces. Communicating each variable when needed. So if you need the vel. in the force computing you can pass it when computing the force and modify the force function accordingly.

*/
#ifndef DRIVER_H
#define DRIVER_H
#include"Interactor/Interactor.h"
#include"Integrator/Integrator.h"
#include"utils/utils.h"

class Driver{
//Integrator and Interactor take care of the
// updating of positions and computing the pair forces. You can do anything in between.
  Interactor *interactor;
  Integrator *integrator;

  uint N;
//You are supposed to be in charge of the positions, and initialize them before giving them to Integrator.
  Vector<float4> pos;
public:
  Driver(uint N, float L, float rcut, float dt);
  
  void update();

  //Write the current positions to disk, concurrently if block is false or not given
  void write(bool block = false);
  //Read an initial configuratio nfrom fileName, TODO
  void read(const char *fileName);

};

#endif

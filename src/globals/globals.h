/*
Raul P. Pelaez 2016. Global parameters for the simulation.

Any module can use these parameters however they want. So the VerletNVT module will use gcnf.T but leave gcnf.E unused. Changing the values of this struct after the simulation has been created will cause strange behaviour.

TODO:
100- Construct the simulation via strings in GlobalConfig
100- Implement a system to broadcast the change of one of the parameters. Maybe something like a global event vector
100- Seed should change in every step acording to a xorshift128 generator, with seed gcnf.seed in the first step, updating it.
 */

#ifndef GLOBALS_H
#define GLOBALS_H
#include "utils/utils.h"
#include "globals/defines.h"

struct GlobalConfig{
  /*Default parameters*/
  uint N = 0;
  real sigma = 1.0;  //Biggest LJ diameter of the particles, wich defines length units
  real3 L = {0.0, 0.0, 0.0};
  real rcut = 2.5;
  real dt = 0.001;
  real T = 0.0;
  real gamma = 1.0; //General damping factor for a thermostat
  real E = 0.0;

  uint nsteps = 0;
  uint nsteps1 = 10000;
  uint nsteps2 = 10000;
  int print_steps=-1;
  int measure_steps = -1;

  ullint seed = 0xf31337Bada55D00d;  
};

#endif

//Everyone can access to gcnf, its initialized in main, before entering main().
extern GlobalConfig gcnf;
//These Arrays are declared in main.cpp and initialized in Driver.cpp
extern Vector4 pos, force;
extern Vector3 vel;
extern Xorshift128plus grng;



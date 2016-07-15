/*
Raul P. Pelaez 2016. Global parameters for the simulation.

Any module can use these parameters however they want. So the VerletNVT module will use gcnf.T but leave gcnf.E unused. Changing the values of this struct after the simulation has been created will cause strange behaviour.

TODO:
100- Construct the simulation via strings in GlobalConfig
100- Implement a system to broadcast the change of one of the parameters. Maybe something like a global event vector
 */
#ifndef GLOBALS_H
#define GLOBALS_H
#include "utils/utils.h"
struct GlobalConfig{
  /*Default parameters*/
  uint N = 16384;
  float L = 32, rcut = 2.5f;
  float dt = 0.001f;
  float T = 0.0f;
  float E = 0.0f;

  uint nsteps = 10000;
  int print_steps=-1;
  uint relaxation_steps = 1000;
  uint measure_steps = -1;

};
#endif

//Everyone can access to gcnf, its initialized in main, before entering main().
extern GlobalConfig gcnf;
//These addresses are set in Driver
extern Vector4 pos, force;
extern Vector3 vel;

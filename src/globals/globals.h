/*
Raul P. Pelaez 2016. Global parameters for the simulation.

Any module can access these parameters however they want. So the VerletNVT module will use gcnf.T but leave gcnf.E unused. Changing the values of this struct after the simulation has been created will cause strange behaviour.

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
  bool D2 = false; /*Two dimensions*/
  real rcut = 2.5;
  real dt = 0.001;
  real T = 0.0;
  real gamma = 1.0; //General damping factor for a thermostat
  real E = 0.0;

  uint nsteps = 0; /*Total steps performed*/
  uint nsteps1 = 0;
  uint nsteps2 = 0;
  int print_steps=-1;
  int measure_steps = -1;
  /*Anytime seed is used, it should be replaced with a new random number*/
  ullint seed = 0xf31337Bada55D00dULL;  
};

/*Usual parameters to use on GPU side, max 64kb(constant memory)*/
struct GlobalConfigGPU{
  bool D2; /*Two dimensions flag*/
  real3 L;
  real3 invL;
  int N;
  real dt;
  real T;
};

#endif
/********************CPU SIDE************/
//Everyone can access to gcnf, its initialized in main, before entering main().
extern GlobalConfig gcnf;

extern uint current_step;
//These Arrays are declared in main.cpp and initialized in Driver.cpp
extern Vector4 pos, force;
extern Vector3 vel;
extern Xorshift128plus grng;

/**************GPU SIDE****************/
extern __constant__ GlobalConfigGPU gcnfGPU;


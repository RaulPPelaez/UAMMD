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
  uint N = 0;         //Number of particles
  real sigma = 1.0;  //Biggest LJ diameter of the particles, wich defines length units
  real3 L = {0.0, 0.0, 0.0};  //Size of the simulation box
  bool D2 = false; /*Two dimensions*/ //Do not manually set this flag.
  real rcut = 2.5;  //A cut-off radius (for PairForces i.e)
  real dt = 0.001;  //Time step size
  real T = 0.0;     //Target Temperature for a constant T ensemble
  real gamma = 1.0; //General damping factor for a thermostat
  real E = 0.0;     //Target energy for a constant E ensemble

  uint nsteps = 0; /*Total steps performed, do not manually change this variable*/
  uint nsteps1 = 0; //Two number of steps, you can use these if you want to run()
  uint nsteps2 = 0;
  int print_steps=-1;     //Print every X steps
  int measure_steps = -1; //Measure every X steps
  /*Anytime seed is used, it should be replaced with a new random number, i.e grng.next()*/
  ullint seed = 0xf31337Bada55D00dULL;

  //names do not have to be adjacent (there can be two colors in the system, 0 and 112 e.g). But color must be adjacent, if there are two types their colors are 0 and 1. This transform between both
  vector<uint> color2name;
  
  
};

/*Usual parameters to use on GPU side, max 64kb(constant memory)*/
struct GlobalConfigGPU{
  bool D2; /*Two dimensions flag*/ //Do not manually set this flag. 
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


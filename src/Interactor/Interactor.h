/*
Raul P. Pelaez 2016. Interactor class GPU wrapper, to use from CPU, handles GPU calls, integration and interaction of particles. Also keeps track of all the variables and writes/reads simulation to disk.

The idea is for this class to be use to compute only the forces, and integrate elsewhere.
Like give me a pointer to positions and get a pointer to forces

See Interactor.cpp for additional info
 */

#ifndef INTERACTOR_H
#define INTERACTOR_H

#include"utils/utils.h"
#include"InteractorGPU.cuh"
#include"misc/Potential.h"
#include<cstdint>

#define fori(x,y) for(int i=x; i<y; i++)
#define forj(x,y) for(int j=x; j<y; j++)

enum forceType{LJ,NONE};

typedef uint32_t uint;
class Interactor{
public:
  Interactor(int N, float L, float rcut,
	     Vector<float4> *d_pos,
	     forceType fs=LJ);
  ~Interactor();

  void compute_force();
  //  float4 *get_force(){return this->force.d_m;}
  Vector<float4>* getForce(){return &force;}
private:
  Vector<float4> *d_pos, force, sortPos, sortForce;
  Vector<float3> vel;
  uint N;
  void init();

  uint ncells;
  Vector<uint> cellIndex, particleIndex; 
  Vector<uint> cellStart, cellEnd;

  float rcut, L, dt;

  InteractorParams params;
  
  forceType forceSelector;

  Potential pot;
};
#endif

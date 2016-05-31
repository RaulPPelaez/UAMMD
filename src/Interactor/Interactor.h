/*
Raul P. Pelaez 2016. Interactor class GPU wrapper, to use from CPU, handles GPU calls, interaction of particles.

The idea is for this class to be use to compute only the forces, and integrate elsewhere.
Like give me a pointer to positions and get a pointer to forces

See Interactor.cpp for additional info

TODO:
90- Springs. Use Floren's algorithm from fluam.
 */

#ifndef INTERACTOR_H
#define INTERACTOR_H

#include"utils/utils.h"
#include"misc/Potential.h"
#include<cstdint>
#include<memory>

class Interactor{
public:
  Interactor();
  ~Interactor();

  virtual void sumForce() = 0;

  shared_ptr<Vector<float4>> getForce(){return force;}

protected:
  Interactor(uint N, float L, 
	     shared_ptr<Vector<float4>> d_pos,
	     shared_ptr<Vector<float4>> force);
  
  shared_ptr<Vector<float4>> d_pos, force;
  uint N;
  float L;
};
#endif

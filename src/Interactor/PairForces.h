/*
Raul P. Pelaez 2016. Short range pair forces Interactor implementation.

  Interactor is intended to be a module that computes and sums the forces acting on each particle
    due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements a Neighbour cell list search and computes the force between all neighbouring
  particle pairs. 
  The Neighbour search is implemented using hash short with cell index as hash.
  
  Force is evaluated using table lookups (with texture memory)
  
TODO:
100- Colors, this almost done, color can simply be encoded in pos.w, and additional parameters are needed in force fucntions/textures
90- Non cubic boxes, almost done, just be carefull in the constructor and use vector types.
100- PairForces should be a singleton or multiple PairForces should be possible somehow

 */

#ifndef PAIRFORCES_H
#define PAIRFORCES_H

#include"utils/utils.h"
#include"Interactor.h"
#include"PairForcesGPU.cuh"
#include"misc/Potential.h"

#include<cstdint>
#include<memory>
#include<functional>

//The currently implemented forces, custom allows for an arbitrary function tu be used as force function
enum pairForceType{LJ,NONE,CUSTOM};

float forceLJ(float r2);
float nullForce(float r2);


class PairForces: public Interactor{
public:
  PairForces(uint N, float L, float rcut,
	     shared_ptr<Vector<float4>> d_pos,
	     shared_ptr<Vector<float4>> force,
	     pairForceType fs=LJ,
	     std::function<float(float)> customForceFunction = nullForce);
  ~PairForces();

  void sumForce() override;
private:
  Vector<float4> sortPos;
  void init();

  uint ncells;
  Vector<uint> cellIndex, particleIndex; 
  Vector<uint> cellStart, cellEnd;

  float rcut;

  PairForcesParams params;
  
  pairForceType forceSelector;

  Potential pot;
  std::function<float(float)> customForceFunction;
};
#endif

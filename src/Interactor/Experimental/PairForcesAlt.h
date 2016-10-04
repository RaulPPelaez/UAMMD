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
80- Change the handling of the potential to better allow inheritance, see DPD
 */

#ifndef PAIRFORCESALT_H
#define PAIRFORCESALT_H
#include"Interactor/PairForces.h"
#include"utils/utils.h"
#include"globals/globals.h"
#include"Interactor/Interactor.h"
#include"PairForcesAltGPU.cuh"
#include"misc/Potential.h"

#include<cstdint>
#include<memory>
#include<functional>

// //The currently implemented forces, custom allows for an arbitrary function tu be used as force function
// enum pairForceType{LJ,NONE,CUSTOM};

// float forceLJ(float r2);
// float energyLJ(float r2);
// float nullForce(float r2);



class PairForcesAlt: public Interactor{
public:
  //PairForcesAlt(pairForceType fs = LJ);
  PairForcesAlt(pairForceType fs = LJ);
  ~PairForcesAlt();

  void sumForce() override;
  float sumEnergy() override;
  float sumVirial() override;
  
protected:
  uint ncells;
  
  void init();
  void makeNeighbourList();
  
 
  pair_forces_alt_ns::Params params;
  

  Vector4 old_pos, sortPos;
  Vector<uint> NBL, Nneigh, CELL;
  Vector<uint> cellIndex, cellSize, particleIndex;
  
  
  Potential pot;
  
  float rcut;
  uint maxNPerCell;
  
  pairForceType forceSelector;
  
  static uint pairForcesInstances;
};
#endif

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
#include"globals/globals.h"
#include"Interactor.h"
#include"PairForcesGPU.cuh"
#include"misc/Potential.h"

#include<cstdint>
#include<memory>
#include<functional>

//The currently implemented forces, custom allows for an arbitrary function tu be used as force function
enum pairForceType{LJ,NONE,CUSTOM};

float forceLJ(float r2);
float energyLJ(float r2);
float nullForce(float r2);


class PairForces: public Interactor{
public:
  PairForces(pairForceType fs = LJ);
  PairForces(pairForceType fs,
	     std::function<float(float)> customForceFunction,
  	     std::function<float(float)> customEnergyFunction);
  ~PairForces();

  void sumForce() override;
  float sumEnergy() override;
  float sumVirial() override;
  
private:
  Vector4 sortPos;
  Vector<float> energyArray, virialArray;
  
  void init();
  void makeNeighbourList();
  
  uint ncells;
  Vector<uint> cellIndex, particleIndex; 
  Vector<uint> cellStart, cellEnd;

  float rcut;

  
  PairForcesParams params;
  
  //These handle the selected force functions
  pairForceType forceSelector;
  Potential pot;
  std::function<float(float)> customForceFunction;
  std::function<float(float)> customEnergyFunction;
};
#endif

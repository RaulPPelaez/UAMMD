/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs

  Interactor is intended to be a module that computes and sums the forces acting on each particle
  due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements an algorithm to compute the force between particles joined by springs.
  Sends the bondList to the GPU ordered by the first particle, and two additional arrays
    storing where the information for each particle begins and ends. Identical to the sorting
    trick in PairForces

  The format of the input file is the following, 
    just give a list of springs, the order doesnt matter as long as no spring is repeated:
   
    i j r0 k
    .
    .
    .
    
    Where i,j are the indices of the particles, r0 the equilibrium distance and k the spring constant

TODO:
100- Allow for fluam like input (AKA check for repeated springs) ignore and print a message.
100- Implement Three body springs
*/

#ifndef BONDEDFORCES_H
#define BONDEDFORCES_H

#include"utils/utils.h"
#include"Interactor.h"
#include"BondedForcesGPU.cuh"
#include"misc/Potential.h"
#include"globals/globals.h"

#include<cstdint>
#include<memory>
#include<functional>
#include<vector>

class BondedForces: public Interactor{
public:
  BondedForces(const std::vector<Bond> &bondList);
  BondedForces(const char * readFile);

  ~BondedForces();

  void sumForce() override;
  float sumEnergy() override;
  float sumVirial() override;
private:
  void init();
  
  uint nbonds;
  Vector<Bond> bondList;
  Vector<uint> bondStart, bondEnd;

  uint nbondsFP; //Fixed Point
  Vector<BondFP> bondListFP;
  Vector<uint> bondStartFP, bondEndFP;

  BondedForcesParams params;
};
#endif

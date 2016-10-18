/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs GPU callers

  Interactor is intended to be a module that computes and sums the forces acting on each particle
  due to some interaction, like and external potential or a pair potential.

  This module implements an algorithm to compute the force between particles joined by springs.
*/


#ifndef BONDEDFORCESGPU_CUH
#define BONDEDFORCESGPU_CUH
#include"globals/defines.h"
namespace bonded_forces_ns{
  struct Bond{
    int i,j;
    real r0,k;
  };

  struct BondFP{
    int i;
    real3 pos;
    real r0,k;
  };

  struct ThreeBond{
    int i,j,k;
    real r0,kspring,ang;
  };


  struct Params{
    real3 L, invL;
  };


  //Stores some simulation parameters to upload as constant memory.
  void initGPU(Params m_params);

  void computeBondedForce(real4 *force, real4 *pos,
			  uint *bondStart, uint *bondEnd, uint *bondedParticleIndex, 
			  Bond* bondList, uint N, uint Nparticles_with_bonds, uint nbonds);

  void computeBondedForceFixedPoint(real4* force, real4 *pos,
				    uint *bondStartFP, uint *bondEndFP, BondFP* bondListFP, uint N, uint nbonds);


  void computeThreeBondedForce(real4 *force, real4 *pos,
			       uint *bondStart, uint *bondEnd, uint *bondedParticleIndex, 
			       ThreeBond* bondList, uint N, uint Nparticles_with_bonds, uint nbonds);
}

#endif









/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs GPU callers

  Interactor is intended to be a module that computes and sums the forces acting on each particle
  due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements an algorithm to compute the force between particles joined by springs.
*/


#ifndef BONDEDFORCESGPU_CUH
#define BONDEDFORCESGPU_CUH

struct Bond{
  int i,j;
  float r0,k;
};

struct BondedForcesParams{
  float L, invL;
};


//Stores some simulation parameters to upload as constant memory.
void initBondedForcesGPU(uint *bondStart, uint *bondEnd, Bond* bondList,
			 uint N, uint nbonds, BondedForcesParams m_params);


void computeBondedForce(float4* force, float4 *pos,
			uint *bondStart, uint *bondEnd, Bond* bondList, uint N, uint nbonds);


#endif









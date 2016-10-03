/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs GPU callers

  Interactor is intended to be a module that computes and sums the forces acting on each particle
  due to some interaction, like and external potential or a pair potential.

  This module implements an algorithm to compute the force between particles joined by springs.
*/


#ifndef BONDEDFORCESGPU_CUH
#define BONDEDFORCESGPU_CUH
namespace bonded_forces_ns{
  struct Bond{
    int i,j;
    float r0,k;
  };

  struct BondFP{
    int i;
    float3 pos;
    float r0,k;
  };

  struct ThreeBond{
    int i,j,k;
    float r0,kspring,ang;
  };


  struct Params{
    float L, invL;
  };


  //Stores some simulation parameters to upload as constant memory.
  void initBondedForcesGPU(Params m_params);

  void computeBondedForce(float4 *force, float4 *pos,
			  uint *bondStart, uint *bondEnd, uint *bondedParticleIndex, 
			  Bond* bondList, uint N, uint Nparticles_with_bonds, uint nbonds);

  void computeBondedForceFixedPoint(float4* force, float4 *pos,
				    uint *bondStartFP, uint *bondEndFP, BondFP* bondListFP, uint N, uint nbonds);


  void computeThreeBondedForce(float4 *force, float4 *pos,
			       uint *bondStart, uint *bondEnd, uint *bondedParticleIndex, 
			       ThreeBond* bondList, uint N, uint Nparticles_with_bonds, uint nbonds);
}

#endif









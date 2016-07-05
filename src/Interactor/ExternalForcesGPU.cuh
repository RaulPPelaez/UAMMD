/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs GPU callers

  Interactor is intended to be a module that computes and sums the forces acting on each particle
  due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements an algorithm to compute the force between particles joined by springs.
*/


#ifndef EXTERNALFORCESGPU_CUH
#define EXTERNALFORCESGPU_CUH

struct ExternalForcesParams{
  float L, invL;
};


//Stores some simulation parameters to upload as constant memory.
void initExternalForcesGPU(ExternalForcesParams m_params);


void computeExternalForce(float4* force, float4 *pos, uint N);

#endif









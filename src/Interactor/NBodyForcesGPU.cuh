/*
  Raul P. Pelaez 2016. NBody Force Interactor GPU kernel callers

  Computes the interaction between all pairs in the system. Currently only gravitational force

*/

#ifndef NBODYFORCESGPU_CUH
#define NBODYFORCESGPU_CUH

void computeNBodyForce(float4* force, float4 *pos, uint N);


#endif









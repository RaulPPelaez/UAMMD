/*
Raul P. Pelaez 2016.

TODO:
100- Colors, this almost done, color can simply be encoded in pos.w, and additional parameters are needed in force fucntions/textures
90- Non cubic boxes, almost done, just be carefull in the constructor and use vector types.
100- PairForces should be a singleton or multiple PairForces should be possible somehow

NOTES:
Use update isntead of update_development to increase performance (update doesnt record time and doesnt sync device)

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
  Vector<float3> vel;
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

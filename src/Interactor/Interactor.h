/*Raul P. Pelaez 2016. Interactor abstract class base implementation

  Interactor is intended to be a module that computes and sums the forces acting on each particle
    due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This is an abstract class that should be derived to implement new interactors, Interactor itself cannot be instanciated.

 Currently Implemented interactors:
   1. Pair short range Forces using neighbour lists for an arbitrary potential

TODO:
90- Springs. Use Floren's algorithm from fluam.
 */

#ifndef INTERACTOR_H
#define INTERACTOR_H

#include"utils/utils.h"
#include"misc/Potential.h"
#include<cstdint>
#include<memory>

class Interactor{
public:
  Interactor();
  ~Interactor();

  virtual void sumForce() = 0;

  shared_ptr<Vector<float4>> getForce(){return force;}

protected:
  Interactor(uint N, float L, 
	     shared_ptr<Vector<float4>> d_pos,
	     shared_ptr<Vector<float4>> force);
  
  shared_ptr<Vector<float4>> d_pos, force;
  uint N;
  float L;
};
#endif

/*Raul P. Pelaez 2016. Interactor abstract class base implementation

  Interactor is intended to be a module that computes and sums the forces acting on each particle
    due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This is an abstract class that should be derived to implement new interactors, Interactor itself cannot be instanciated.

 Currently Implemented interactors:
   1. Pair short range Forces using neighbour lists for an arbitrary potential
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
  virtual float sumEnergy() = 0;
  virtual float sumVirial() = 0;

  Vector4Ptr getForce(){return force;}

protected:
  Interactor(uint N, float L, 
	     Vector4Ptr d_pos,
	     Vector4Ptr force);
  
  Vector4Ptr d_pos, force;
  uint N;
  float L;
};
typedef vector<shared_ptr<Interactor>> InteractorArray;
#endif

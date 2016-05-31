/*Raul P. Pelaez 2016. Interactor abstract class base implementation

  Interactor is intended to be a module that computes and sums the forces acting on each particle
    due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This is an abstract class that should be derived to implement new interactors, Interactor itself cannot be instanciated.

 Currently Implemented interactors:
   1. Pair short range Forces using neighbour lists for an arbitrary potential
 
*/


#include"Interactor.h"

Interactor::Interactor(){}

Interactor::~Interactor(){}


Interactor::Interactor(uint N, float L, 
		       shared_ptr<Vector<float4>> d_pos,
		       shared_ptr<Vector<float4>> force):
  N(N),L(L), d_pos(d_pos), force(force){


}

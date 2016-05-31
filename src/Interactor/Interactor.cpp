


#include"Interactor.h"

Interactor::Interactor(){}

Interactor::~Interactor(){}


Interactor::Interactor(uint N, float L, 
		       shared_ptr<Vector<float4>> d_pos,
		       shared_ptr<Vector<float4>> force):
  N(N),L(L), d_pos(d_pos), force(force){


}

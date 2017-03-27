
#ifndef PAIRFORCESDPD_H
#define PAIRFORCESDPD_H

#include"globals/defines.h"
#include"PairForces.h"


class PairForcesDPD: public PairForces{
public:
  PairForcesDPD();
  ~PairForcesDPD(){}
  
  void makeNeighbourListDPD();
  
  void sumForce() override;
  real sumEnergy() override;
  real sumVirial() override;
  
private:
  real gamma;
  Vector4 sortVel;
  
  pair_forces_ns::ParamsDPD paramsDPD;
  unsigned long long int seed;
};

#endif

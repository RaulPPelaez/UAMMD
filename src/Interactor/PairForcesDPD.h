
#ifndef PAIRFORCESDPD_H
#define PAIRFORCESDPD_H

#include"PairForces.h"


class PairForcesDPD: public PairForces{
public:
  PairForcesDPD();
  ~PairForcesDPD(){}
  
  void makeNeighbourListDPD();
  
  void sumForce() override;
  float sumEnergy() override;
  float sumVirial() override;
  
private:
  float gamma;
  Vector4 sortVel;
  
  pair_forces_ns::ParamsDPD paramsDPD;
  Xorshift128plus rngCPU;
  unsigned long long int seed;
};

#endif

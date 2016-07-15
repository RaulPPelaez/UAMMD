
#ifndef PAIRFORCESDPD_H
#define PAIRFORCESDPD_H

#include"PairForces.h"

class PairForcesDPD: public PairForces{
public:
  PairForcesDPD();
  ~PairForcesDPD(){}

  void sumForce() override;
  float sumEnergy() override;
  float sumVirial() override;
  
private:
  float gamma;

  PairForcesParamsDPD paramsDPD;
};




#endif

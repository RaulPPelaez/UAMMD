/*
Raul P. Pelaez 2016. EnergyMeasure Measurable class implementation.

Computes the total, internal and kinetic energy and virial pressure of the system.
Writes the results to measurables.dat

*/

#ifndef ENERGYMEASURE_H
#define ENERGYMEASURE_H
#include"utils/utils.h"
#include"globals/defines.h"
#include"globals/globals.h"
#include"Interactor/Interactor.h"
#include"Integrator/Integrator.h"
#include"Measurable.h"

class EnergyMeasure: public Measurable{
public:
  EnergyMeasure(InteractorArray interactors,
		shared_ptr<Integrator> integrator);
  ~EnergyMeasure();

  void measure() override;


  operator shared_ptr<EnergyMeasure>(){return make_shared<EnergyMeasure>(*this);}
private:
  //Needs to know all the interactors and integrators in the system.
  InteractorArray interactors;
  shared_ptr<Integrator> integrator;
  uint step;  
  real K, U, P, rho;

};


#endif

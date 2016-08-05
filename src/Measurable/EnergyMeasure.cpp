/*
Raul P. Pelaez 2016. EnergyMeasure Measurable class implementation.

Computes the total, internal and kinetic energy and virial pressure of the system.
Writes the results to measurables.dat

*/

#include"EnergyMeasure.h"

EnergyMeasure::EnergyMeasure(InteractorArray interactors,
			     shared_ptr<Integrator> integrator):
  Measurable(),
  interactors(interactors),
  integrator(integrator),
  step(0),
  K(0.0f), U(0.0f), P(0.0f){
  
  rho = N/(L*L*L);

  //Each measurable should print a header informing of what it is going to print
  out<<"#E\tK\tU\t\tP";

  /*All measurables are printed on the same line and file each step, you should include these lines in
   constructor of your implementation*/
  if(id==total_measurables) out<<"\n";
  else out<<"\t\t";
      
}
EnergyMeasure::~EnergyMeasure(){}

void EnergyMeasure::measure(){
  step++;
  K = 0.0f;
  U = 0.0f;
  P = 0.0f;

  /*Compute kinetic energy per particle*/
  K = integrator->sumEnergy();
  
  float T = 2.0f*K/3.0f; //Temperature
  
  for(auto i: interactors){
    U += i->sumEnergy(); //Compute potential energy
    P += i->sumVirial(); //Compute virial pressure
  }
  P = rho*T-P;

  
  
  out<<(U+K)<<"\t"<<K<<"\t"<<U<<"\t"<<P; //Print your measurables to out
  /*Include this two lines in your measurable*/
  if(id==total_measurables) out<<endl; //All measurables are printed on the same line and file each step
  else out<<"\t\t";
}

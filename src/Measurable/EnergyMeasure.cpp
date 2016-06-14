/*
Raul P. Pelaez 2016. EnergyMeasure Measurable class implementation.

Computes the total, internal and kinetic energy and virial pressure of the system.
Writes the results to measurables.dat

*/

#include"EnergyMeasure.h"

EnergyMeasure::EnergyMeasure(InteractorArray interactors,
			     shared_ptr<Integrator> integrator, uint N, float L):
  interactors(interactors),
  integrator(integrator),
  Measurable()
{
  
  step = 0;
  K = U = 0.0f;

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

  /*Compute kinetic energy*/
  K = integrator->sumEnergy();
  
  float T = K/3.0f; //Temperature
  
  for(auto i: interactors){
    U += i->sumEnergy(); //Compute potential energy
    P += rho*T+i->sumVirial(); //Compute virial pressure
  }
    

  
  
  out<<(U+K)<<"\t"<<K<<"\t"<<U<<"\t"<<P; //Print your measurables to out

  /*Include this two lines in your measurable*/
  if(id==total_measurables) out<<"\n"; //All measurables are printed on the same line and file each step
  else out<<"\t\t";
}

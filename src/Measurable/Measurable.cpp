/*
Raul P. Pelaez 2016. Measurable abstract class.

A measurable is any computation that has to be done during steps of the simulation.
This can be anything like computing the internal energy, the radial function distribution or any
arbitrary calculation on the simulation variables. Although it is possible, measurable should
not  directly change in any way any of the simulation variables, like the position or the force.

A Measurable only has the method measure, which performs the computation and decides what to do with it.
The idea is for each measurable to print to measurables.dat each time measure is called.
Check EnergyMeasure for an example. 

Each Measurable has an id that can be used to know the order of the calls.

*/

#include "Measurable.h"


int Measurable::total_measurables = -1;
ofstream Measurable::out;


Measurable::Measurable(){
  total_measurables++;
  this->id = total_measurables;
  this-> N = gcnf.N;
  this-> L = gcnf.L;
  if(total_measurables==0){
     out.open("measurable.dat");
  }
}

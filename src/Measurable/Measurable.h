/*
Raul P. Pelaez 2016. Measurable abstract class.

A measurable is any computation that has to be done during steps of the simulation.
This can be anything like computing the internal energy, the radial function distribution or any
arbitrary calculation on the simulation variables. Although it is possible, measurable should
not  directly change in any way any of the simulation variables, like the position or the force.

A Measurable only has the method measure, which performs the computation and decides what to do with it.
The idiea is for each measurable to print to measurables.dat each time measure is called.
Check EnergyMeasure for an example. 

Each Measurable has an id that can be used to know the order of the calls.
*/

#ifndef MEASURABLE_H
#define MEASURABLE_H
#include "utils/utils.h"
#include"globals/defines.h"
#include"globals/globals.h"
#include<fstream>

class Measurable{
public:
  Measurable();
  ~Measurable(){}
  virtual void measure() = 0;
  
protected:
  uint N;
  real3 L;
  int id;
  static std::ofstream out;
  static int total_measurables;
};
typedef std::vector<shared_ptr<Measurable>> MeasurableArray;

#endif

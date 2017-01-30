/* WORK IN PROGRESS
  Raul P. Pelaez 2016. SimulationScript class definition.

  This class inherits from Driver. 
  Reads the simulation configuration from a script and runs it.


TODO: 
100- Find a way to map a string to a new pointer of an interactor/integrator type. smt like


template<typename T> 
struct count_arg;

template<typename R, typename ...Args> 
struct count_arg<std::function<R(Args...)>>
{
    static const size_t value = sizeof...(Args);
};

template<class T>
 shared_ptr<T> CreateInstanceIntegrator(string args){
 return make_shared<T>(args);
}

 map<string, std::function<shared_ptr<Integrator>(string)>> integratorMap;
 interactorMap["PairForces"] = &CreateInstanceIntegrator<PairForces>;
.
.
.
integrator->addinteractor( interactorMap[option](args) );

100- Find a way to properly handle arguments to each module.
*/


#ifndef SIMULATIONSCRIPT_H
#define SIMULATIONSCRIPT_H

#include"globals/defines.h"
#include "Driver.h"
#include<map>
#include<string>
#include<memory>
#include<vector>

class SimulationScript: public Driver{
public:
  SimulationScript(int argc, char* argv[], const char *fileName);
  ~SimulationScript(){}

protected:
  std::map<std::string, shared_ptr<Interactor>> interactorMap;
  std::map<std::string, shared_ptr<Integrator>> integratorMap;
  std::map<std::string, double> parameterMap;

  std::vector<shared_ptr<Interactor>> interactors;
  std::vector<shared_ptr<Integrator>> integrators;
  
  void setUpMaps();
  
};



#endif

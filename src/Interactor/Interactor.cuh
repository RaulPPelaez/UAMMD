/*Raul P. Pelaez 2017. Interactor Base class.

Interactor is an interface for modules that can compute forces and energies.
For a class to be a valid Interactor, it must override sumForces() and sumEnergy().

An integrator will expect interactors to describe the interaction of particles.

Interactor is also ParameterUpdatable, which means that any Interactor can override any of the available update*() functions.

See the following wiki pages for more info:
ParameterUpdatable
Interactor
Integrator

See PairForces.cuh or examples/LJ.cu for an example.

 */

#ifndef INTERACTOR_CUH
#define INTERACTOR_CUH

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"

#include<memory>
#include<vector>
#include"third_party/type_names.h"
#include"misc/ParameterUpdatable.h"
namespace uammd{
  class Interactor: public virtual ParameterUpdatable{
  protected:
    string name;
    shared_ptr<ParticleData> pd;
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;
  
  public:

    Interactor(shared_ptr<ParticleData> pd,	       
	       shared_ptr<System> sys,
	       std::string name="noName"):
      Interactor(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, name){}

    Interactor(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       std::string name="noName"):
      pd(pd), pg(pg), sys(sys), name(name){
      sys->log<System::MESSAGE>("[Interactor] %s created.", name.c_str());
      sys->log<System::MESSAGE>("[Interactor] Acting on group %s", pg->getName().c_str());
    }

    ~Interactor(){
      sys->log<System::DEBUG>("[Interactor] %s Destroyed", name.c_str());
    }
    
    virtual void sumForce(cudaStream_t st) = 0;
    virtual real sumEnergy(){ return 0;}

    std::string getName(){ return this->name;}

};

}

#endif
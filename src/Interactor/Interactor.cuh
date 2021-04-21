/*Raul P. Pelaez 2017-2021. Interactor Base class.

Interactor is an interface for modules that can compute forces, energies or perform an arbitrary computation according to a certain interaction.
For a class to be a valid Interactor, it must override sumForces() and sumEnergy().

An integrator will expect interactors to describe the interaction of particles.

Interactor is also ParameterUpdatable, which means that any Interactor can override any of the available update*() functions.

See the following wiki pages for more info:
ParameterUpdatable
Interactor
Integrator

 */

#ifndef INTERACTOR_CUH
#define INTERACTOR_CUH

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"

#include<memory>
#include <stdexcept>
#include<vector>
#include"third_party/type_names.h"
#include"misc/ParameterUpdatable.h"
namespace uammd{
  struct Computables{
    bool force = false;
    bool energy = false;
    bool virial = false;
  };
  
  class Interactor: public virtual ParameterUpdatable{
  protected:
    string name;
    shared_ptr<ParticleData> pd;
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;
    virtual ~Interactor(){
      sys->log<System::DEBUG>("[Interactor] %s Destroyed", name.c_str());
    }

  public:

    Interactor(shared_ptr<ParticleData> pd, std::string name="noName"):
      Interactor(pd, std::make_shared<ParticleGroup>(pd, "All"), pd->getSystem(), name){}

    Interactor(shared_ptr<ParticleData> pd, shared_ptr<ParticleGroup> pg,
	       std::string name="noName"):
      Interactor(pd, pg, pd->getSystem(), name){}

    Interactor(shared_ptr<ParticleData> pd,
	       shared_ptr<System> sys,
	       std::string name="noName"):
      Interactor(pd, std::make_shared<ParticleGroup>(pd, "All"), sys, name){}

    Interactor(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> i_sys,
	       std::string name="noName"):
      pd(pd), pg(pg), sys(i_sys), name(name){
      if(i_sys != pd->getSystem()){
	sys->log<System::EXCEPTION>("[Interactor] Cannot work with a different System than ParticleData");
	throw std::invalid_argument("[Interactor] Invalid System");
      }
      sys->log<System::MESSAGE>("[Interactor] %s created.", name.c_str());
      sys->log<System::MESSAGE>("[Interactor] Acting on group %s", pg->getName().c_str());
    }

    //This function must compute the forces due to the particular interaction and add them to pd->getForces().
    //For that it can make use of the provided CUDA stream
    virtual void sumForce(cudaStream_t st = 0) = 0;

    //This function must compute the energies due to the particular interaction and add them to pd->getEnergy()
    //The return value is unused
    virtual real sumEnergy(){ return 0;}

    //This function can be defined to compute force and energy at the same time.
    //By default it will sequentially call sumForce and them sumEnergy.
    virtual real sumForceEnergy(cudaStream_t st = 0){
      sumForce(st);
      return sumEnergy();
    }

    virtual void sum(Computables comp, cudaStream_t st){
      if(comp.force and comp.energy){
	sumForceEnergy(st);
      }
      else if (comp.force){
	sumForce(st);
      }
      else if(comp.energy){
	sumEnergy();
      }
    }
    
    //The compute function can perform any arbitrary computation without any warranties about what it does.
    //Look in the wiki page for each Interactor for information on how to use this function.
    virtual void compute(cudaStream_t st = 0){

    }

    std::string getName(){return this->name;}

};

}

#endif

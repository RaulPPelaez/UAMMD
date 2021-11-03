/*Raul P. Pelaez 2017-2021. Interactor Base class.

Interactor is an interface for modules that can compute forces, energies and/or virials according to a certain interaction.
For a class to be a valid Interactor, it must override sum() member function.

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
#include"third_party/type_names.h"
#include"misc/ParameterUpdatable.h"
namespace uammd{

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
      Interactor(std::make_shared<ParticleGroup>(pd, "All"), name){}

    Interactor(shared_ptr<ParticleGroup> pg, std::string name="noName"):
      pd(pg->getParticleData()), pg(pg), sys(pd->getSystem()), name(name){
      sys->log<System::MESSAGE>("[Interactor] %s created.", name.c_str());
      sys->log<System::MESSAGE>("[Interactor] Acting on group %s", pg->getName().c_str());
    }

    //This struct exposes the different targets of computation that can be requested from an Interactor.
    struct Computables{
      bool force = false;
      bool energy = false;
      bool virial = false;
    };

    virtual void sum(Computables comp, cudaStream_t st = 0) = 0;
    
    std::string getName(){return this->name;}

};

}

#endif

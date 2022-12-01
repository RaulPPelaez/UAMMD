/*Raul P. Pelaez 2017-2022. Interactor Base class.

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

// Extra computables can be defined at compile time, for instance for new
// extensions.
//Example:
//#define EXTRA_COMPUTABLES (mycomputable1)(mycomputable2)
#ifndef EXTRA_COMPUTABLES
#define EXTRA_COMPUTABLES
#endif

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include<memory>
#include"third_party/type_names.h"
#include "misc/ParameterUpdatable.h"
#include <third_party/boost/preprocessor.hpp>
#include<third_party/boost/preprocessor/seq/for_each.hpp>
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
      bool force  = false;
      bool energy = false;
      bool virial = false;
      bool stress = false;
#define DECLARE_EXTRA_COMPUTABLES(r,data,name) bool name = false;
      BOOST_PP_SEQ_FOR_EACH(DECLARE_EXTRA_COMPUTABLES, _, EXTRA_COMPUTABLES)
#undef DECLARE_EXTRA_COMPUTABLES
    };

    virtual void sum(Computables comp, cudaStream_t st = 0) = 0;

    std::string getName(){return this->name;}

};

}

#endif

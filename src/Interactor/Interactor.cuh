

#ifndef INTERACTOR_CUH
#define INTERACTOR_CUH

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"

#include<memory>
#include<vector>
#include"third_party/type_names.h"
#include"misc/ParameterUpdatable.h"
namespace uammd{
  class Interactor: public ParameterUpdatable{
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
    virtual real sumEnergy() = 0;

    std::string getName(){ return this->name;}

};

}

#endif


#ifndef INTERACTOR_CUH
#define INTERACTOR_CUH

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"

#include<memory>
#include<vector>

namespace uammd{
  class Interactor{
  protected:
    string name;
    shared_ptr<ParticleData> pd;
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;
  
  public:
    
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
    virtual void compute(){}
    virtual void sumForce(cudaStream_t st) = 0;
    virtual real sumEnergy() = 0;

    std::string getName(){ return this->name;}

};

}

#endif
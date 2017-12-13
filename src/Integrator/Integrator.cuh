/* Raul P. Pelaez 2017. Integrator Base class

   An integrator has the ability to move the simulation one step forward in time.
   
   For that, it can use any number of interactors.

Integrator is an UAMMD base module, 
to create an Integrator module inherit from Integrator and overload the virtual methods.


 */
#ifndef INTEGRATOR_CUH
#define INTEGRATOR_CUH

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"Interactor/Interactor.cuh"
#include<memory>
#include<vector>

namespace uammd{
  class Integrator{
  protected:
    string name;
    shared_ptr<ParticleData> pd;
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;
  
    std::vector<shared_ptr<Interactor>> interactors;
  public:

    Integrator(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       std::string name="noName"):
      pd(pd), pg(pg), sys(sys), name(name){
      sys->log<System::MESSAGE>("[Integrator] %s created.", name.c_str());
      sys->log<System::MESSAGE>("[Integrator] Acting on group %s", pg->getName().c_str());
    }

    ~Integrator(){
      sys->log<System::DEBUG>("[Integrator] %s Destroyed", name.c_str());
    }
    
    virtual void forwardTime() = 0;
    virtual real sumEnergy(){ return 0.0;}


    //The interactors can be called at any time from the integrator to compute the forces when needed.
    void addInteractor(shared_ptr<Interactor> an_interactor){
      sys->log<System::MESSAGE>("[%s] Adding Interactor %s...", name.c_str(), an_interactor->getName().c_str());
      interactors.emplace_back(an_interactor);      
    }
    
    std::vector<std::shared_ptr<Interactor>> getInteractors(){
      return interactors;
    }            
        
};

}

#endif
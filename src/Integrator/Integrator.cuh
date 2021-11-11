/* Raul P. Pelaez 2017-2021. Integrator Base class
   Integrator is an UAMMD base module,
   to create an Integrator module inherit from Integrator and overload the virtual methods.
   An Integrator has the ability to move the simulation one step forward in time.

   For that, it can use any number of Interactors.
   
   Integrator can also hold a list of references to ParameterUpdatable derived objects.
   Adding an Interactor via addInteractor() will also add it as a ParameterUpdatable.

   See the related wiki page for more information.

   USAGE:
   //Say "integrator" is an instance of an Integrator derived class (such as BD::EulerMaruyama or VerletNVE).
   
   //Forward the simulation to the next time step:
   integrator.forwardTime();
   
   //Add an Interactor derived object (as a shared_ptr) to the integrator:
   integrator.addInteractor(an_interactor);
   //Add a ParameterUpdatable derived object (asa shared_ptr) to the integrator:
   //Note that interactors are automatically added as updatables as well, so there is no need to manually add them as updatables.
   integrator.addUpdatable(an_updatable);   

   //Get a list of Interactors that have been added to the Integrator:
   auto interactors = integrator.getInteractors();
   //Get a list of ParameterUpdatables that have been added to the Integrator (note that this includes Interactors):
   auto updatables = integrator.getUpdatables();
   //Note that calling any "update" method for both the interactors and updatables lists will duplicate calls to the update methods of the interactors.
   //If the Integrator needs to update parameters it should do so using only the list provided by addUpdatables().



 */
#ifndef INTEGRATOR_CUH
#define INTEGRATOR_CUH

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
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
    std::set<shared_ptr<ParameterUpdatable>> updatables;
    virtual ~Integrator(){
      sys->log<System::DEBUG>("[Integrator] %s Destroyed", name.c_str());
    }

  public:

    Integrator(shared_ptr<ParticleData> pd, std::string name="noName"):
      Integrator(pd, std::make_shared<ParticleGroup>(pd, "All"), pd->getSystem(), name){}

    Integrator(shared_ptr<ParticleData> pd, shared_ptr<ParticleGroup> pg,
	       std::string name="noName"):
      Integrator(pd, pg, pd->getSystem(), name){}

    Integrator(shared_ptr<ParticleData> pd,
	       shared_ptr<System> sys,
	       std::string name="noName"):
      Integrator(pd, std::make_shared<ParticleGroup>(pd, "All"), sys, name){}

    Integrator(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> i_sys,
	       std::string name="noName"):
      pd(pd), pg(pg), sys(i_sys), name(name){
      if(i_sys != pd->getSystem()){
	sys->log<System::EXCEPTION>("[Integrator] Cannot work with a different System than ParticleData");
	throw std::invalid_argument("[Integrator] Invalid System");
      }
      sys->log<System::MESSAGE>("[Integrator] %s created.", name.c_str());
      sys->log<System::MESSAGE>("[Integrator] Acting on group %s", pg->getName().c_str());
    }

    virtual void forwardTime() = 0;

    virtual real sumEnergy(){ return 0.0;}

    //Add an Interactor to the Integrator.
    //This also adds it as an updatable, so there is no need to also call addUpdatable for Interactors.
    void addInteractor(shared_ptr<Interactor> an_interactor){
      sys->log<System::MESSAGE>("[%s] Adding Interactor %s...", name.c_str(), an_interactor->getName().c_str());
      interactors.emplace_back(an_interactor);
      this->addUpdatable(an_interactor);
    }

    std::vector<std::shared_ptr<Interactor>> getInteractors(){
      return interactors;
    }

    //Adds a ParameterUpdatable to the Integrator.    
    void addUpdatable(shared_ptr<ParameterUpdatable> an_updatable){
      sys->log<System::MESSAGE>("[%s] Adding updatable", name.c_str());
      updatables.emplace(an_updatable);
    }

    std::vector<std::shared_ptr<ParameterUpdatable>> getUpdatables(){
      return {updatables.begin(), updatables.end()};
    }

};

}

#endif

/*Pablo Palacios-Alonso 2024. DPD integrator module
  
  This module simulates particle dynamics using a Dissipative Particle Dynamics (DPD) simulation.
  To achieve this, it employs the Verlet NVE algorithm to update the positions of the particles
  in conjunction with an interactor that treats the typical DPD forces [1] as a UAMMD potential.


  References:
  [1] On the numerical treatment of dissipative particle dynamics and related systems. Leimkuhler and Shang 2015. https://doi.org/10.1016/j.jcp.2014.09.008
*/

#pragma once

#include "Integrator/VerletNVE.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/Potential/DPD.cuh"

namespace uammd{
  class DPDIntegrator: public Integrator{
    std::shared_ptr<VerletNVE> verlet;
    int steps;
  public:
    struct Parameters{
      //VerletNVE
      real dt = 0;
      bool is2D = false;
            
      //DPD
      real cutOff;
      real gamma;
      real A;
      real temperature;
      Box box;
    };
    
    DPDIntegrator(shared_ptr<ParticleGroup> pg, Parameters par):
      Integrator(pg, "DPDIntgrator"),
      steps(0){
      //Initialize verletNVE
      VerletNVE::Parameters verletpar;
      verletpar.dt             = par.dt;
      verletpar.is2D           = par.is2D;
      verletpar.initVelocities = false;
      
      verlet = std::make_shared<VerletNVE>(pg, verletpar);      
      
      //Initialize DPD and add to interactor list.
      Potential::DPD::Parameters dpdPar;
      dpdPar.temperature    = par.temperature;
      dpdPar.cutOff         = par.cutOff;
      dpdPar.gamma          = par.gamma;
      dpdPar.A              = par.A;
      dpdPar.dt             = par.dt;
      auto dpd = std::make_shared<Potential::DPD>(dpdPar);
      using PF = PairForces<Potential::DPD>;
      typename PF::Parameters pfpar;
      pfpar.box = par.box;
      //From the example in PairForces
      auto dpd_interactor = std::make_shared<PF>(pg, pfpar, dpd);
      verlet->addInteractor(dpd_interactor);
    }
    
    DPDIntegrator(shared_ptr<ParticleData> pd, Parameters par):
      DPDIntegrator(std::make_shared<ParticleGroup>(pd, "All"), par){}
    
    ~DPDIntegrator(){}
    
    virtual void forwardTime() override {
      steps++;
      if (steps == 1){
	for(auto forceComp: interactors){
	  verlet->addInteractor(forceComp);
	}
      }
      verlet->forwardTime();
    }
    
    virtual real sumEnergy() override {
      return verlet->sumEnergy();      
    }
    
  };
}

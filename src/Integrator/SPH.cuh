/*Pablo Palacios-Alonso 2024. SPH integrator module
  
  This module simulates particle dynamics using a Dissipative Particle Dynamics (DPD) simulation.
  To achieve this, it employs the Verlet NVE algorithm to update the positions of the particles
  in conjunction with an interactor that treats the typical DPD forces [1] as a UAMMD potential.


  References:
  [1] On the numerical treatment of dissipative particle dynamics and related systems. Leimkuhler and Shang 2015. https://doi.org/10.1016/j.jcp.2014.09.008
*/

#pragma once

#include "Integrator/VerletNVE.cuh"
#include "Interactor/SPH.cuh"

namespace uammd{
  class SPHIntegrator: public Integrator{
    std::shared_ptr<VerletNVE> verlet;
    int steps;
  public:
    struct Parameters{
      using NeighbourList = VerletList;
      //VerletNVE
      real dt = 0;
      bool is2D = false;
      bool initVelocities = true; //Modify initial particle velocities to enforce the provided energy
      real energy = 0; //Target energy, ignored if initVelocities is false
      real mass = -1;
      //SPH
      Box box;
      real support = 1.0; 
      real viscosity = 50.0;
      real gasStiffness = 100.0;
      real restDensity = 0.4;
      std::shared_ptr<NeighbourList> nl = nullptr;
    };
    
    SPHIntegrator(shared_ptr<ParticleGroup> pg, Parameters par):
      Integrator(pg, "SPHIntgrator"),
      steps(0){
      //Initialize verletNVE
      VerletNVE::Parameters verletpar;
      verletpar.dt             = par.dt;
      verletpar.is2D           = par.is2D;
      verletpar.initVelocities = par.initVelocities;
      verletpar.energy         = par.energy;
      
      
      verlet = std::make_shared<VerletNVE>(pg, verletpar);      
      
      //Initialize SPH and add to interactor list.
      SPH::Parameters sphPar;
      sphPar.support      = par.support;
      sphPar.gasStiffness = par.gasStiffness;
      sphPar.restDensity  = par.restDensity;
      sphPar.viscosity    = par.viscosity;
      sphPar.box          = par.box;
      
      auto sph = std::make_shared<SPH>(pd, sphPar);
      verlet->addInteractor(sph);
    }
    
    SPHIntegrator(shared_ptr<ParticleData> pd, Parameters par):
      SPHIntegrator(std::make_shared<ParticleGroup>(pd, "All"), par){}
    
    ~SPHIntegrator(){}
    
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

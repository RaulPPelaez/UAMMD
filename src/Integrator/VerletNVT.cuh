/*Raul P. Pelaez 2017-2021. Verlet NVT Integrator module.

  This module integrates the dynamic of the particles using a two step velocity verlet MD algorithm
  that conserves the temperature, volume and number of particles.

  For that several thermostats are implemented:

    - Basic (Velocity damping and gaussian noise force)
    - Gronbech Jensen [1]
    - BBK ( TODO)
    - SPV( TODO)
 Usage:

    Create the module as any other integrator with the following parameters:

    auto sys = make_shared<System>();
    auto pd = make_shared<ParticleData>(N,sys);
    auto pg = make_shared<ParticleGroup>(pd,sys, "All");

    using NVT = VerletNVT::GronbechJensen;
    NVT::Parameters par;
     par.temperature = 1.0;
     par.dt = 0.01;
     par.viscosity = 1.0;
     par.is2D = false;
     //If mass is specified all particles will be assumed to have this mass. If unspecified pd::getMass will be used, if it has not been requested, all particles are assumed to have mass=1.
     //par.mass = 1.0;
    auto verlet = make_shared<NVT>(pd, pg, sys, par);

    //Add any interactor
    verlet->addInteractor(...);
    ...

    //forward simulation 1 dt:

    verlet->forwardTime();

-----
References:

[1] N. Gronbech-Jensen, and O. Farago: "A simple and effective Verlet-type
algorithm for simulating Langevin dynamics", Molecular Physics (2013).
http://dx.doi.org/10.1080/00268976.2012.760055


 */
#ifndef VERLETNVT_CUH
#define VERLETNVT_CUH

#include "Integrator/Integrator.cuh"

namespace uammd{
  namespace VerletNVT{
    class Basic: public Integrator{
    public:
      struct Parameters{
	real temperature = 0;
	real dt = 0;
	real viscosity = 1.0;
	bool is2D = false;
	real mass = -1.0;
      };
    protected:
      real noiseAmplitude;
      uint seed;
      real dt, temperature, viscosity;
      real defaultMass;
      bool is2D;

      cudaStream_t stream;
      int steps;

      //Constructor for derived classes
      Basic(shared_ptr<ParticleData> pd,
	    shared_ptr<ParticleGroup> pg,
	    shared_ptr<System> sys,
	    Parameters par,
	    std::string name);
    public:
      Basic(shared_ptr<ParticleData> pd,
	    shared_ptr<ParticleGroup> pg,
	    shared_ptr<System> sys,
	    Parameters par);
      Basic(shared_ptr<ParticleData> pd,
	    shared_ptr<System> sys,
	    Parameters par):
	Basic(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, par){}

      virtual ~Basic();

      virtual void forwardTime() override;
      virtual real sumEnergy() override{ return 0;};
    };


    class GronbechJensen final: public Basic{
    public:
      GronbechJensen(shared_ptr<ParticleData> pd,
		     shared_ptr<ParticleGroup> pg,
		     shared_ptr<System> sys,
		     Basic::Parameters par):
	Basic(pd, pg, sys, par, "VerletNVT::GronbechJensen"){}

      GronbechJensen(shared_ptr<ParticleData> pd,
		     shared_ptr<System> sys,
		     Basic::Parameters par):
	Basic(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, par, "VerletNVT::GronbechJensen"){}

      ~GronbechJensen(){}

      using Parameters = Basic::Parameters;

      virtual void forwardTime() override;
    private:
      template<int step> void callIntegrate();
    };


  }
}

#include"VerletNVT/Basic.cu"
#include"VerletNVT/GronbechJensen.cu"
#endif

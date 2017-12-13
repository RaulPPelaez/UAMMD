/*Raul P. Pelaez 2017. Verlet NVT Integrator module.

  This module integrates the dynamic of the particles using a two step velocity verlet MD algorithm
  that conserves the temperature, volume and number of particles.

  For that several thermostats are (should be, currently only one) implemented:

    - Basic (Velocity damping and gaussian noise)
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
#include <curand.h>
#include<thrust/device_vector.h>
namespace uammd{
  namespace VerletNVT{
    class Basic: public Integrator{
    protected:
      real noiseAmplitude;
      real dt, temperature, viscosity;    
      bool is2D;
      curandGenerator_t curng;
      thrust::device_vector<real3> noise;

    
      cudaStream_t forceStream, stream;
      cudaEvent_t forceEvent;
      int steps;
      
      void genNoise(cudaStream_t st);
    public:
      struct Parameters{
	real temperature = 0;
	real dt = 0;
	real viscosity = 1.0;
	bool is2D = false;
      };
      Basic(shared_ptr<ParticleData> pd,
	    shared_ptr<ParticleGroup> pg,
	    shared_ptr<System> sys,
	    Parameters par);
      ~Basic();

      virtual void forwardTime() override;
      virtual real sumEnergy() override{ return 0;};
    };

    
    class GronbechJensen: public Basic{
    public:
      using Basic::Basic;
      using Parameters = Basic::Parameters;

      virtual void forwardTime() override;
      
    };


  }
}

#include"VerletNVT/Basic.cu"
#include"VerletNVT/GronbechJensen.cu"
#endif
  
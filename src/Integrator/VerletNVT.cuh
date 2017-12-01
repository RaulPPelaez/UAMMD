/*Raul P. Pelaez 2017. Verlet NVT Integrator module.

  This module integrates the dynamic of the particles using a two step velocity verlet MD algorithm
  that conserves the temperature, volume and number of particles.

  For that several thermostats are (should be, currently only one) implemented:

    -Velocity damping and gaussian noise 
    - BBK ( TODO)
    - SPV( TODO)
 Usage:
 
    Create the module as any other integrator with the following parameters:
    
    
    auto sys = make_shared<System>();
    auto pd = make_shared<ParticleData>(N,sys);
    auto pg = make_shared<ParticleGroup>(pd,sys, "All");
    
    
    VerletNVT::Parameters par;
     par.temperature = 1.0;
     par.dt = 0.01;
     par.damping = 1.0;
     par.is2D = false;

    auto verlet = make_shared<VerletNVT>(pd, pg, sys, par);
      
    //Add any interactor
    verlet->addInteractor(...);
    ...
    
    //forward simulation 1 dt:
    
    verlet->forwardTime();
    
TODO:

100- Outsource thermostat logic to a functor (external or internal)

 */
#ifndef VERLETNVT_CUH
#define VERLETNVT_CUH

#include "Integrator/Integrator.cuh"
#include <curand.h>
#include<thrust/device_vector.h>
namespace uammd{
  class VerletNVT: public Integrator{
    real noiseAmplitude;
    real dt, temperature, damping;    
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
      real damping = 1.0;
      bool is2D = false;
    };
    VerletNVT(shared_ptr<ParticleData> pd,
	      shared_ptr<ParticleGroup> pg,
	      shared_ptr<System> sys,
	      Parameters par);
    ~VerletNVT();

    virtual void forwardTime() override;
    virtual real sumEnergy() override{ return 0;};
  };

}

#include"VerletNVT.cu"
#endif
  
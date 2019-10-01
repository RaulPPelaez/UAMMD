/*Raul P. Pelaez 2017. Verlet NVE Integrator module.

  This module integrates the dynamic of the particles using a two step velocity verlet MD algorithm
  that conserves the energy, volume and number of particles.

  Create the module as any other integrator with the following parameters:


  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N,sys);
  auto pg = make_shared<ParticleGroup>(pd,sys, "All");


  VerletNVE::Parameters par;
     par.energy = 1.0; //Target energy per particle, can be ommited if initVelocities=false
     par.dt = 0.01;
     par.is2D = false;
     par.initVelocities=true; //Modify starting velocities to ensure the target energy

    auto verlet = make_shared<VerletNVE>(pd, pg, sys, par);

    //Add any interactor
    verlet->addInteractor(...);
    ...

    //forward simulation 1 dt:

    verlet->forwardTime();

 */
#ifndef VERLETNVE_CUH
#define VERLETNVE_CUH

#include "Integrator/Integrator.cuh"
#include <curand.h>
#include<thrust/device_vector.h>
namespace uammd{
  class VerletNVE: public Integrator{
    real dt;
    real energy;
    bool is2D;
    bool initVelocities;

    cudaStream_t stream;
    int steps;

    size_t temp_storage_bytes;
    
  public:
    struct Parameters{
      real energy = 0;
      real dt = 0;
      bool is2D = false;
      bool initVelocities = true;
    };
    VerletNVE(shared_ptr<ParticleData> pd,
	      shared_ptr<System> sys,
	      Parameters par):
      VerletNVE(pd, std::make_shared<ParticleGroup>(pd, sys, "All"),sys, par){}

    VerletNVE(shared_ptr<ParticleData> pd,
	      shared_ptr<ParticleGroup> pg,
	      shared_ptr<System> sys,
	      Parameters par);
    ~VerletNVE();

    virtual void forwardTime() override;
    virtual real sumEnergy() override;
  };

}

#include"VerletNVE.cu"
#endif


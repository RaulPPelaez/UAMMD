/*Raul P. Pelaez 2017-2021. Verlet NVT Integrator module.


  This module integrates the dynamic of the particles using a two step velocity verlet MD algorithm
  that conserves the energy, volume and number of particles.

  Create the module as any other integrator with the following parameters:

  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N,sys);
  auto pg = make_shared<ParticleGroup>(pd,sys, "All");

  VerletNVE::Parameters par;
     par.energy = 1.0; //Target energy per particle, will be ignored if initVelocities=false
     par.dt = 0.01;
     par.is2D = false;
     par.initVelocities=true; //Modify starting velocities to ensure the target energy, default is true

    auto verlet = make_shared<VerletNVE>(pd, pg, sys, par);

    //Add any interactor
    verlet->addInteractor(...);
    ...

    //forward simulation 1 dt:

    verlet->forwardTime();

 */

#include"VerletNVE.cuh"


#ifndef SINGLE_PRECISION
#define curandGenerateNormal curandGenerateNormalDouble
#endif

namespace uammd{

  VerletNVE::VerletNVE(shared_ptr<ParticleData> pd,
		       shared_ptr<ParticleGroup> pg,
		       shared_ptr<System> sys,
		       VerletNVE::Parameters par):
    Integrator(pd, pg, sys, "VerletNVE"),
    dt(par.dt), energy(par.energy), is2D(par.is2D), initVelocities(par.initVelocities),
    steps(0){
    if(initVelocities)
      sys->log<System::MESSAGE>("[VerletNVE] Target energy per particle: %g", energy);
    else
      sys->log<System::MESSAGE>("[VerletNVE] Not fixing an initial per particle energy.");

    sys->log<System::MESSAGE>("[VerletNVE] Time step: %g", dt);
    if(is2D){
      sys->log<System::MESSAGE>("[VerletNVE] Working in 2D mode.");
    }
    int numberParticles = pg->getNumberParticles();
    if(pd->isVelAllocated() && initVelocities){
      sys->log<System::WARNING>("[VerletNVE] Velocity will be overwritten to ensure energy conservation!");
    }
    bool useDefaultMass = not pd->isMassAllocated();
    this->defaultMass = par.mass;
    if(useDefaultMass and this->defaultMass < 0 ){
      this->defaultMass = 1.0;
    }
    CudaSafeCall(cudaStreamCreate(&stream));
  }

  VerletNVE::~VerletNVE(){
    cudaStreamDestroy(stream);
  }

  namespace VerletNVE_ns{

    //Integrate the movement 0.5 dt and reset the forces in the first step
    template<int step>
      __global__ void integrateGPU(real4* __restrict__  pos,
				   real3* __restrict__ vel,
				   const real4* __restrict__  force,
				   const real* __restrict__ mass,
				   real defaultMass,
				   ParticleGroup::IndexIterator indexIterator,
				   int N,
				   real dt, bool is2D){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	//Index of current particle in group
	int i = indexIterator[id];
	//Half step velocity
	real m = mass?mass[i]:defaultMass;
	vel[i] += (make_real3(force[i])/m)*dt*real(0.5);
	if(is2D) vel[i].z = real(0.0);
	//In the first step, upload positions
	if(step==1){
	  real3 newPos = make_real3(pos[i]) + vel[i]*dt;
	  pos[i] = make_real4(newPos, pos[i].w);
	}
      }
  }

  void VerletNVE::initializeVelocities(){
    //In the first step, compute energy in the system
    //in order to adapt the initial kinetic energy to match the input total energy
    //E = U+K
    real U = 0.0;
    int numberParticles = pg->getNumberParticles();
    {
      auto energy = pd->getEnergy(access::gpu, access::write);
      auto energy_gr = pg->getPropertyIterator(energy);
      thrust::fill(thrust::cuda::par, energy_gr, energy_gr + numberParticles, real(0.0));
    }
    for(auto forceComp: interactors){
      U += forceComp->sumEnergy();
    }
    {
      auto energy = pd->getEnergy(access::gpu, access::read);
      auto energy_gr = pg->getPropertyIterator(energy);
      U += thrust::reduce(thrust::cuda::par, energy_gr, energy_gr + numberParticles);
    }
    U = U/numberParticles;
    real K = energy - U;
    if(K<0){
      sys->log<System::ERROR>("[VerletNVE] Cannot fix requested energy per particle. Requested E = U + K = %g, but U=%g", energy, U);
      throw std::runtime_error("[VerletNVE] Cannot fix energy");
    }
    sys->log<System::MESSAGE>("Starting potential energy per particle: %g", U);
    auto vel  = pd->getVel(access::location::cpu, access::mode::write);
    auto vel_gr = pg->getPropertyIterator(vel);
    auto mass = pd->getMassIfAllocated(access::location::cpu, access::mode::read).raw();
    auto gindex = pg->getIndexIterator(access::cpu);
    std::transform(gindex, gindex + pg->getNumberParticles(),
		   vel_gr,
		   [&](int i){
		     auto dir = make_real3(sys->rng().gaussian3(0.0,1.0));
		     dir = dir/sqrt(dot(dir, dir));
		     //K = 0.5*m*v^2 -> v = sqrt(2*K/m)
		     real m = mass?mass[i]:defaultMass;
		     return make_real3(sqrt(2.0*K/m)*dir);
		   });
    sys->log<System::MESSAGE>("Starting kinetic energy per particle: %g", K);
  }

  template<int step>
  void VerletNVE::callIntegrate(){
    int numberParticles = pg->getNumberParticles();
    int Nthreads = 128;
    int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
    //An iterator with the global indices of my groups particles
    auto groupIterator = pg->getIndexIterator(access::location::gpu);
    //Get all necessary properties
    auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
    auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
    auto force = pd->getForce(access::location::gpu, access::mode::read);
    auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read).raw();
    //First step integration and reset forces
    VerletNVE_ns::integrateGPU<step><<<Nblocks, Nthreads, 0, stream>>>(pos.raw(),
								       vel.raw(),
								       force.raw(),
								       mass,
								       defaultMass,
								       groupIterator,
								       numberParticles, dt, is2D);
  }

  void VerletNVE::resetForces(){
    int numberParticles = pg->getNumberParticles();
    auto force = pd->getForce(access::location::gpu, access::mode::write);
    auto force_gr = pg->getPropertyIterator(force);
    thrust::fill(thrust::cuda::par.on(stream), force_gr, force_gr + numberParticles, real4());
  }

  void VerletNVE::firstStepPreparation(){
    if(initVelocities){
      initializeVelocities();
    }
    resetForces();
    for(auto forceComp: interactors){
      forceComp->updateTimeStep(dt);
      forceComp->sumForce(stream);
    }
  }

  //Move the particles in my group 1 dt in time.
  void VerletNVE::forwardTime(){
    steps++;
    sys->log<System::DEBUG1>("[VerletNVE] Performing integration step %d", steps);
    if(steps==1)
      firstStepPreparation();
    callIntegrate<1>();
    resetForces();
    for(auto forceComp: interactors){
      forceComp->updateSimulationTime(steps*dt);
      forceComp->sumForce(stream);
    }
    callIntegrate<2>();
  }

  namespace verletnve_ns{
    template<class VelocityIterator, class MassIterator, class EnergyIterator>
    __global__ void sumKineticEnergy(const VelocityIterator vel,
				     EnergyIterator energy,
				     const MassIterator mass,
				     int numberParticles){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id >= numberParticles) return;
      auto v = vel[id];
      energy[id] += real(0.5)*dot(v,v)*mass[id];
    }
  };


  real VerletNVE::sumEnergy(){
    int numberParticles = pg->getNumberParticles();
    auto vel_gr = pg->getPropertyIterator(pd->getVel(access::location::gpu, access::mode::read));
    auto energy_gr = pg->getPropertyIterator(pd->getEnergy(access::location::gpu, access::mode::write));
    int Nthreads = 128;
    int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
    auto mass = pd->getMassIfAllocated(access::gpu, access::read);
    if(defaultMass > 0){
      auto mass_iterator = thrust::make_constant_iterator(defaultMass);
      verletnve_ns::sumKineticEnergy<<<Nblocks, Nthreads>>>(vel_gr, energy_gr, mass_iterator, numberParticles);
    }
    else{
      auto mass_iterator = pg->getPropertyIterator(mass);
      verletnve_ns::sumKineticEnergy<<<Nblocks, Nthreads>>>(vel_gr, energy_gr, mass_iterator, numberParticles);
    }

    return 0.0;
  }

}


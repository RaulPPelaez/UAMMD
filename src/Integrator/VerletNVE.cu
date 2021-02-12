/*Raul P. Pelaez 2017. Verlet NVT Integrator module.


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
      sys->log<System::MESSAGE>("[VerletNVE] Energy: %.3f", energy);
    else
      sys->log<System::MESSAGE>("[VerletNVE] Not enforcing input energy.");

    sys->log<System::MESSAGE>("[VerletNVE] Time step: %.3f", dt);
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
				   real4* __restrict__  force,
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
	real invMass = real(1.0)/defaultMass;
	if(mass){
	  invMass = real(1.0)/mass[i];
	}
	vel[i] += (make_real3(force[i])*invMass)*dt*real(0.5);
	if(is2D) vel[i].z = real(0.0);
	//In the first step, upload positions
	if(step==1){
	  real3 newPos = make_real3(pos[i]) + vel[i]*dt;
	  pos[i] = make_real4(newPos, pos[i].w);
	  //Reset force
	  force[i] = make_real4(0);
	}
      }
  }

  void VerletNVE::initializeVelocities(){
    //In the first step, compute energy in the system
    //in order to adapt the initial kinetic energy to match the input total energy
    //E = U+K
    real U = 0.0;
    for(auto forceComp: interactors) U += forceComp->sumEnergy();
    real K = abs(energy - U);
    real vamp = sqrt(2.0*K/3.0);
    auto vel  = pd->getVel(access::location::cpu, access::mode::write);
    auto vel_gr = pg->getPropertyIterator(vel);
    std::generate(vel_gr, vel_gr + pg->getNumberParticles(), [&](){return make_real3(vamp*sys->rng().gaussian3(0.0, 1.0));});
  }

  template<int step>
  void VerletNVE::callIntegrate(){
    int numberParticles = pg->getNumberParticles();
    int Nthreads=128;
    int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);    
    //An iterator with the global indices of my groups particles
    auto groupIterator = pg->getIndexIterator(access::location::gpu);
    //Get all necessary properties
    auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
    auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
    auto force = pd->getForce(access::location::gpu, access::mode::read);
    auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read).raw();
    if(this->defaultMass > 0){
      mass = nullptr;
    }
    //First step integration and reset forces
    VerletNVE_ns::integrateGPU<step><<<Nblocks, Nthreads, 0, stream>>>(pos.raw(),
								       vel.raw(),
								       force.raw(),
								       mass,
								       defaultMass,
								       groupIterator,
								       numberParticles, dt, is2D);
  }

  //Move the particles in my group 1 dt in time.
  void VerletNVE::forwardTime(){
    steps++;
    sys->log<System::DEBUG1>("[VerletNVE] Performing integration step %d", steps);
    for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*dt);
    if(steps==1){
      if(initVelocities){
	initializeVelocities();
      }
      for(auto forceComp: interactors){
	forceComp->updateTimeStep(dt);
      }
    }
    //First simulation step is special
    if(steps==1){
      {
	int numberParticles = pg->getNumberParticles();
	auto force = pd->getForce(access::location::gpu, access::mode::write);
	auto force_gr = pg->getPropertyIterator(force);
	thrust::fill(thrust::cuda::par.on(stream), force_gr, force_gr + numberParticles, real4());
      }
      for(auto forceComp: interactors){
	forceComp->updateTimeStep(dt);
	forceComp->sumForce(stream);
      }
      cudaDeviceSynchronize();
    }
    callIntegrate<1>();
    for(auto forceComp: interactors){
      forceComp->sumForce(stream);
    }
    callIntegrate<2>();
  }

  namespace VerletNVE_ns{
    __global__ void sumEnergy(const real3* vel,
			      real *energy,
			      const real *mass,
			      real defaultMass,
			      ParticleGroup::IndexIterator groupIterator,
			      int numberParticles){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id >= numberParticles) return;
      const int i = groupIterator[id];
      const real mass_i = mass?mass[i]:defaultMass;
      energy[i] += real(0.5)*dot(vel[i], vel[i])*mass_i;
    }
  };

  real VerletNVE::sumEnergy(){
    int numberParticles = pg->getNumberParticles();
    auto groupIterator = pg->getIndexIterator(access::location::gpu);
    auto vel = pd->getVel(access::location::gpu, access::mode::read);
    auto Energy = pd->getEnergy(access::location::gpu, access::mode::write);
    auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read).raw();
    if(this->defaultMass > 0){
      mass = nullptr;
    }
    int Nthreads = 128;
    int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
    VerletNVE_ns::sumEnergy<<<Nblocks, Nthreads>>>(vel.raw(),
						   Energy.raw(),
						   mass, defaultMass,
						   groupIterator,
						   numberParticles);
    return 0.0;
  }

}







































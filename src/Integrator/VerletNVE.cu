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
    
    cudaStreamCreate(&stream);
  }

  
  VerletNVE::~VerletNVE(){
    cudaStreamDestroy(stream);
    cudaFree(d_K);
    cudaFree(d_tmp_storage);
  }



  namespace VerletNVE_ns{

    //Integrate the movement 0.5 dt and reset the forces in the first step
    template<int step>
      __global__ void integrateGPU(real4* __restrict__  pos,
				   real3* __restrict__ vel,
				   real4* __restrict__  force,
				   const real* __restrict__ mass,
				   ParticleGroup::IndexIterator __restrict__ indexIterator,
				   int N,
				   real dt, bool is2D){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	//Index of current particle in group
	int i = indexIterator[id];
	
	//Half step velocity
	real invMass = real(1.0);
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
  
  
  //Move the particles in my group 1 dt in time.
  void VerletNVE::forwardTime(){
    steps++;
    sys->log<System::DEBUG1>("[VerletNVE] Performing integration step %d", steps);
    for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*dt);
    
    int numberParticles = pg->getNumberParticles();
    if(steps==1){
      if(initVelocities){
	//In the first step, compute the force and energy in the system
	//in order to adapt the initial kinetic energy to match the input total energy
	//E = U+K 
	real U = 0.0;
	for(auto forceComp: interactors) U += forceComp->sumEnergy();
	real K = abs(energy - U);
	//Distribute the velocities accordingly
	real vamp = sqrt(2.0*K/3.0);
	//Create velocities
	auto vel  = pd->getVel(access::location::cpu, access::mode::write);
	auto groupIterator = pg->getIndexIterator(access::location::cpu);
	forj(0, numberParticles){
	  int i = groupIterator[j];
	  vel.raw()[i].x = vamp*sys->rng().gaussian(0.0, 1.0);
	  vel.raw()[i].y = vamp*sys->rng().gaussian(0.0, 1.0);
	  vel.raw()[i].z = vamp*sys->rng().gaussian(0.0, 1.0);
	}
      }
      for(auto forceComp: interactors) 	forceComp->updateTimeStep(dt);      
    }
  

    
    int Nthreads=128;
    int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);

    
    //First simulation step is special
    if(steps==1){
      {
	auto groupIterator = pg->getIndexIterator(access::location::gpu);
	auto force = pd->getForce(access::location::gpu, access::mode::write);     
	fillWithGPU<<<Nblocks, Nthreads>>>(force.raw(), groupIterator, make_real4(0), numberParticles);
      }
      for(auto forceComp: interactors) forceComp->sumForce(stream);
      cudaDeviceSynchronize();
    }
    


    
    //First integration step
    {
      //An iterator with the global indices of my groups particles
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      //Get all necessary properties
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
      auto force = pd->getForce(access::location::gpu, access::mode::read);     
      //Mass is assumed 1 for all particles if it has not been set.
      auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);
      //First step integration and reset forces

      VerletNVE_ns::integrateGPU<1><<<Nblocks, Nthreads, 0, stream>>>(pos.raw(),
									   vel.raw(),
									   force.raw(),
								      mass.raw(),
									   groupIterator,
									   numberParticles, dt, is2D);
    }

    for(auto forceComp: interactors) forceComp->sumForce(stream);
    //Second integration step
    {
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
            
      auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);
      //Wait untill all forces have been summed
      VerletNVE_ns::integrateGPU<2><<<Nblocks, Nthreads, 0 , stream>>>(pos.raw(),
									    vel.raw(),
									    force.raw(),
								       mass.raw(),
									    groupIterator,
									    numberParticles, dt, is2D);      
    }
    
  }





  namespace VerletNVE_ns{
    __global__ void sumEnergy(real3* vel,
			      real *Energy,
			      real *mass,
			      ParticleGroup::IndexIterator groupIterator,
			      int numberParticles){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id >= numberParticles) return;
      const int i = groupIterator[id];
      real mass_i = real(1.0);
      if(mass) mass_i = mass[i];
      
      Energy[i] += real(0.5)*dot(vel[i], vel[i])*mass_i;
      

    }
  };

  real VerletNVE::sumEnergy(){  
    int numberParticles = pg->getNumberParticles();
    auto groupIterator = pg->getIndexIterator(access::location::gpu);
    
    auto vel = pd->getVel(access::location::gpu, access::mode::read);
    auto Energy = pd->getEnergy(access::location::gpu, access::mode::write);
    
    real * mass_ptr = nullptr;
    
    if(pd->isMassAllocated()){
      auto mass = pd->getMass(access::location::gpu, access::mode::read);
      mass_ptr = mass.raw();
    }
    int Nthreads = 128;
    int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
    
    VerletNVE_ns::sumEnergy<<<Nblocks, Nthreads>>>(vel.raw(),
						   Energy.raw(),
						   mass_ptr,
						   groupIterator,
						   numberParticles);        
    return 0.0;
  }

}







































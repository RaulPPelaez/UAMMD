/*Raul P. Pelaez 2017. Verlet NVT Integrator module.

  This module integrates the dynamic of the particles using a two step velocity verlet MD algorithm
  that conserves the temperature, volume and number of particles.
  
  Uses a simple thermostat with velocity damping and a gaussian noise force.

 Usage:
 
    Create the module as any other integrator with the following parameters:
    
    
    auto sys = make_shared<System>();
    auto pd = make_shared<ParticleData>(N,sys);
    auto pg = make_shared<ParticleGroup>(pd,sys, "All");
    
    using NVT = VerletNVT::Basic;
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
    
 */

#include"../VerletNVT.cuh"

#include"third_party/saruprng.cuh"



namespace uammd{

  namespace VerletNVT{
    namespace Basic_ns{
      //Fill the initial velocities of the particles in my group with a gaussian distribution according with my temperature.
      __global__ void initialVelocities(real3* vel, const real* mass,
					ParticleGroup::IndexIterator indexIterator, //global index of particles in my group
					real vamp, bool is2D, int N, uint seed){
	int id = blockIdx.x*blockDim.x + threadIdx.x;      
	if(id>=N) return;
	Saru rng(id, seed);
	int i = indexIterator[id];
	
	real mass_i = real(1.0);
	if(mass) mass_i = mass[i];
	
	double3 noisei = make_double3(rng.gd(0, vamp/mass_i), rng.gd(0, vamp/mass_i).x); //noise[id];
	
	int index = indexIterator[i];
	vel[index].x = noisei.x;
	vel[index].y = noisei.y;
	if(!is2D){
	  vel[index].z = noisei.z;
	}
      }
    
    }

    Basic::Basic(shared_ptr<ParticleData> pd,
		 shared_ptr<ParticleGroup> pg,
		 shared_ptr<System> sys,		       
		 Basic::Parameters par):
      Basic(pd, pg, sys, par, "VerletNVT::Basic"){}
    
      Basic::Basic(shared_ptr<ParticleData> pd,
		   shared_ptr<ParticleGroup> pg,
		   shared_ptr<System> sys,		       
		   Basic::Parameters par,
		   std::string name):
      Integrator(pd, pg, sys, name),
      dt(par.dt), temperature(par.temperature), viscosity(par.viscosity), is2D(par.is2D),
      steps(0){

      sys->rng().next32();
      sys->rng().next32();
      seed = sys->rng().next32();
      sys->log<System::MESSAGE>("[%s] Temperature: %f", name.c_str(), temperature);
      sys->log<System::MESSAGE>("[%s] Time step: %f", name.c_str(), dt);
      sys->log<System::MESSAGE>("[%s] Viscosity: %f", name.c_str(), viscosity);
      if(is2D){
	sys->log<System::MESSAGE>("[%s] Working in 2D mode.", name.c_str());
      }

      this->noiseAmplitude = sqrt(2*dt*6*M_PI*viscosity*temperature);

      int numberParticles = pg->getNumberParticles();

      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);

      if(pd->isVelAllocated()){
	sys->log<System::WARNING>("[%s] Velocity will be overwritten to ensure temperature conservation!", name.c_str());
      }
      {
	auto vel_handle = pd->getVel(access::location::gpu, access::mode::write);
	auto groupIterator = pg->getIndexIterator(access::location::gpu);
      
	real velAmplitude = sqrt(3.0*temperature);
      	
	auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);

      
	Basic_ns::initialVelocities<<<Nblocks, Nthreads>>>(vel_handle.raw(),
							   mass.raw(),
							   groupIterator,
							   velAmplitude, is2D, numberParticles,
							   sys->rng().next32());
      
      }

      cudaStreamCreate(&stream);

    }


  
    Basic::~Basic(){
      cudaStreamDestroy(stream);
    }



    namespace Basic_ns{

      //Integrate the movement 1 dt and reset the forces in the first step
      template<int step>
      __global__ void integrateGPU(real4* __restrict__  pos,
				   real3* __restrict__ vel,
				   real4* __restrict__  force,
				   const real* __restrict__ mass,
				   const real* __restrict__ radius,
				   ParticleGroup::IndexIterator indexIterator,
				   int N,
				   real dt, real viscosity, bool is2D,
				   real noiseAmplitude,
				   uint stepNum, uint seed){
	const int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	//Index of current particle in group
	const int i = indexIterator[id];

	real invMass = real(1.0);
	if(mass){
	  invMass = real(1.0)/mass[i];
	}
	real radius_i = real(1.0);
	if(radius){
	  radius_i = radius[i];
	}
	
	Saru rng(id, stepNum, seed);
	
	noiseAmplitude *= sqrtf(0.5*radius_i*invMass);
	
	real3 noisei = make_real3(rng.gf(0, noiseAmplitude), rng.gf(0, noiseAmplitude).x); //noise[id];
	
	const real damping = real(6.0)*real(M_PI)*viscosity*radius_i;

	vel[i] += (make_real3(force[i])-damping*vel[i])*(dt*real(0.5)*invMass) + noisei;
	
	if(is2D) vel[i].z = real(0.0);

	//In the first step, upload positions
	if(step==1){
	  const real3 newPos = make_real3(pos[i]) + vel[i]*dt;
	  pos[i] = make_real4(newPos, pos[i].w);
	  //Reset force
	  force[i] = make_real4(0);
	}

      }


    }
  
    //Move the particles in my group 1 dt in time.
    void Basic::forwardTime(){
      for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*dt);
    
      steps++;
      sys->log<System::DEBUG1>("[%s] Performing integration step %d", name.c_str(), steps);
    
      int numberParticles = pg->getNumberParticles();
    
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);

    
      //First simulation step is special
      if(steps==1){
	{
	  auto groupIterator = pg->getIndexIterator(access::location::gpu);
	  auto force = pd->getForce(access::location::gpu, access::mode::write);     
	  fillWithGPU<<<Nblocks, Nthreads>>>(force.raw(), groupIterator, make_real4(0), numberParticles);
	}
	for(auto forceComp: interactors){
	  forceComp->updateTemperature(temperature);
	  forceComp->updateTimeStep(dt);
	  forceComp->sumForce(stream);
	}
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
	auto radius = pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);

	/*First step integration and reset forces*/

	Basic_ns::integrateGPU<1><<<Nblocks, Nthreads, 0, stream>>>(pos.raw(),
								    vel.raw(),
								    force.raw(),
								    mass.raw(),
								    radius.raw(),
								    groupIterator,
								    numberParticles, dt, viscosity, is2D,
								    noiseAmplitude,
								    steps, seed);
      }
      //Compute all the forces
      for(auto forceComp: interactors) forceComp->sumForce(stream);

    
      //Second integration step
      {
	auto groupIterator = pg->getIndexIterator(access::location::gpu);
      
	auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
	auto force = pd->getForce(access::location::gpu, access::mode::read);
      
	auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);
	auto radius = pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);     

	Basic_ns::integrateGPU<2><<<Nblocks, Nthreads, 0 , stream>>>(pos.raw(),
								     vel.raw(),
								     force.raw(),
								     mass.raw(),
								     radius.raw(),
								     groupIterator,
								     numberParticles, dt, viscosity, is2D,
								     noiseAmplitude,
								     steps, seed);
      }

    }
  }
}

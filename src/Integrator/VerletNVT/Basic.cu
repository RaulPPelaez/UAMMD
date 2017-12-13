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
     par.viscosity = 1.0;
     par.is2D = false;

    auto verlet = make_shared<VerletNVT>(pd, pg, sys, par);
      
    //Add any interactor
    verlet->addInteractor(...);
    ...
    
    //forward simulation 1 dt:
    
    verlet->forwardTime();
    
TODO:

100- Outsource thermostat logic to a functor (external or internal)
100-Implement thermostat from https://arxiv.org/pdf/1212.1244.pdf
 */

#include"../VerletNVT.cuh"

#ifndef SINGLE_PRECISION

#define curandGenerateNormal curandGenerateNormalDouble
#endif


namespace uammd{

  namespace VerletNVT{
    namespace Basic_ns{
      //Fill the initial velocities of the particles in my group with a gaussian distribution according with my temperature.
      __global__ void initialVelocities(real3* vel, const real* mass, const real3* noise,
					ParticleGroup::IndexIterator indexIterator, //global index of particles in my group
					real vamp, bool is2D, int N){
	int id = blockIdx.x*blockDim.x + threadIdx.x;      
	if(id>=N) return;
	int i = indexIterator[id];
      
	real mass_i = real(1.0);
	if(mass) mass_i = mass[i];
	int index = indexIterator[i];
	vel[index].x = vamp*noise[i].x/mass_i;
	vel[index].y = vamp*noise[i].y/mass_i;
	if(!is2D){
	  vel[index].z = vamp*noise[i].z/mass_i;
	}
      }
    
    }
  
    Basic::Basic(shared_ptr<ParticleData> pd,
		 shared_ptr<ParticleGroup> pg,
		 shared_ptr<System> sys,		       
		 Basic::Parameters par):
      Integrator(pd, pg, sys, "VerletNVT::Basic"),
      dt(par.dt), temperature(par.temperature), viscosity(par.viscosity), is2D(par.is2D),
      steps(0){
    
      sys->log<System::MESSAGE>("[%s] Temperature: %f", name.c_str(), temperature);
      sys->log<System::MESSAGE>("[%s] Time step: %f", name.c_str(), dt);
      sys->log<System::MESSAGE>("[%s] Viscosity: %f", name.c_str(), viscosity);
      if(is2D){
	sys->log<System::MESSAGE>("[%s] Working in 2D mode.", name.c_str());
      }

      this->noiseAmplitude = sqrt(dt*6*M_PI*viscosity*temperature);

      //Init rng
      curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
    
      curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next());
    
      int numberParticles = pg->getNumberParticles();
      noise.resize(numberParticles);

      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);

           
      //This shit is obscure, curand will only work with an even number of elements
      real* noise_ptr = (real *) thrust::raw_pointer_cast(noise.data());
      //Warm cuRNG
      curandGenerateNormal(curng, noise_ptr, 3*noise.size(), 0.0, 1.0);
      curandGenerateNormal(curng, noise_ptr, 3*noise.size(), 0.0, 1.0);

      if(pd->isVelAllocated()){
	sys->log<System::WARNING>("[%s] Velocity will be overwritten to ensure temperature conservation!", name.c_str());
      }
      {
	auto vel_handle = pd->getVel(access::location::gpu, access::mode::write);
	auto groupIterator = pg->getIndexIterator(access::location::gpu);
      
	real velAmplitude = sqrt(3.0*temperature);
      
	auto noise_ptr = thrust::raw_pointer_cast(noise.data());
	real * mass_ptr = nullptr;
	if(pd->isMassAllocated()){
	  auto mass = pd->getMass(access::location::gpu, access::mode::read);
	  mass_ptr = mass.raw();
	}
      
	Basic_ns::initialVelocities<<<Nblocks, Nthreads>>>(vel_handle.raw(),
							   mass_ptr,
							   noise_ptr,
							   groupIterator,
							   velAmplitude, is2D, numberParticles);
	curandGenerateNormal(curng, (real*)noise_ptr, 3*numberParticles + ((3*numberParticles)%2), 0.0, 1.0);
      
      }

      cudaStreamCreate(&stream);
      cudaStreamCreate(&forceStream);
      cudaEventCreate(&forceEvent);
      //This line makes the code go much slower, I do not know why    
      //cudaEventCreateWithFlags(&forceEvent, cudaEventDisableTiming);
    }


  
    Basic::~Basic(){
      curandDestroyGenerator(curng);
      cudaStreamDestroy(stream);
      cudaEventDestroy(forceEvent);
    }



    namespace Basic_ns{

      //Integrate the movement 1 dt and reset the forces in the first step
      template<int step>
      __global__ void integrateGPU(real4 __restrict__  *pos,
				   real3 __restrict__ *vel,
				   real4 __restrict__  *force,
				   const real __restrict__ *mass,
				   const real __restrict__ *radius,
				   const real3 __restrict__ *noise,
				   ParticleGroup::IndexIterator indexIterator,
				   int N,
				   real dt, real viscosity, bool is2D){
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
	const real damping = real(6.0)*real(M_PI)*viscosity*radius_i;

	vel[i] += (make_real3(force[i])-damping*vel[i])*(dt*real(0.5)*invMass) + noise[id]*sqrtf(radius_i*invMass);
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
  

    //Fill noise array with a gaussian distribution with mean 0 and std noiseAmplitude
    void Basic::genNoise(cudaStream_t st){

      real * noise_ptr = (real *) thrust::raw_pointer_cast(noise.data());
      curandSetStream(curng, st);
      curandGenerateNormal(curng, (real*) noise_ptr,
			   3*noise.size(),
			   real(0.0), noiseAmplitude);
    }
  
    //Move the particles in my group 1 dt in time.
    void Basic::forwardTime(){
      for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*dt);
    
      steps++;
      sys->log<System::DEBUG1>("[%s] Performing integration step %d", name.c_str(), steps);
    
      int numberParticles = pg->getNumberParticles();
      //Handle if the number of particles in my group has changed
      if(noise.size() != numberParticles)  noise.resize(numberParticles);

    
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
	  forceComp->sumForce(forceStream);
	}
	/*Gen noise*/
	genNoise(stream);
	cudaDeviceSynchronize();
      }
      genNoise(stream);
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

	//Second half of noise vector is used for first integration step
	auto noise_ptr = thrust::raw_pointer_cast(noise.data());
      
	/*First step integration and reset forces*/

	Basic_ns::integrateGPU<1><<<Nblocks, Nthreads, 0, stream>>>(pos.raw(),
								    vel.raw(),
								    force.raw(),
								    mass.raw(),
								    radius.raw(),
								    noise_ptr,
								    groupIterator,
								    numberParticles, dt, viscosity, is2D);
      }
      //Gen noise and compute forces at the same time
      cudaEventRecord(forceEvent, stream);
      //Gen noise for two integration steps at once
      genNoise(stream);
      //Compute all the forces
      cudaStreamWaitEvent(forceStream, forceEvent, 0);
      for(auto forceComp: interactors) forceComp->sumForce(forceStream);
      cudaEventRecord(forceEvent, forceStream);
    
      //Second integration step
      {
	auto groupIterator = pg->getIndexIterator(access::location::gpu);
      
	auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
	auto force = pd->getForce(access::location::gpu, access::mode::read);
      
	auto noise_ptr = thrust::raw_pointer_cast(noise.data());
	
	auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);
	auto radius = pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);
      
	//Wait untill all forces have been summed
	cudaStreamWaitEvent(stream, forceEvent, 0);
	Basic_ns::integrateGPU<2><<<Nblocks, Nthreads, 0 , stream>>>(pos.raw(),
								     vel.raw(),
								     force.raw(),
								     mass.raw(),
								     radius.raw(),
								     noise_ptr,
								     groupIterator,
								     numberParticles, dt, viscosity, is2D);      
      }

    }
  }
}

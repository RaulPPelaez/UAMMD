
/*Raul P. Pelaez 2017. Verlet NVT Integrator module.

  This module integrates the dynamic of the particles using a two step velocity verlet MD algorithm
  that conserves the temperature, volume and number of particles.
  
  The algorithm implemented is GronbechJensen[1]
 Usage:
 
    Create the module as any other integrator with the following parameters:
    
    
    auto sys = make_shared<System>();
    auto pd = make_shared<ParticleData>(N,sys);
    auto pg = make_shared<ParticleGroup>(pd,sys, "All");
    
    using NVT = VerletNVT::GronbechJensen;
    NVT::Parameters par;
     par.temperature = 1.0;
     par.dt = 0.01;
     par.damping = 1.0;
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

#include"../VerletNVT.cuh"

#ifndef SINGLE_PRECISION
#define curandGenerateNormal curandGenerateNormalDouble
#endif


namespace uammd{
  namespace VerletNVT{
    namespace GronbechJensen_ns{

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

	if(step==1){
	  real b = real(1.0)/(real(1.0) + damping*dt*invMass*real(0.5));
	
	  real a = (real(1.0)-damping*dt*real(0.5)*invMass)*b;
       
	
	  real3 p = make_real3(pos[i]);
	  p = p +
	    b*dt*vel[i] +
	    b*dt*dt*real(0.5)*invMass*make_real3(force[i]) +
	    b*dt*real(0.5)*invMass*sqrtf(2.0f*radius_i)*noise[id];
	
	  pos[i] = make_real4(p, pos[i].w);

	  vel[i] = a*vel[i] +
	    dt*real(0.5)*invMass*a*make_real3(force[i]) +
	    b*invMass*sqrtf(2.0f*radius_i)*noise[id];
	  
	  if(is2D) vel[i].z = real(0.0);
	  
	  force[i] = make_real4(0);
	}      
	else{
	  vel[i] += dt*real(0.5)*invMass*make_real3(force[i]);
	}

      }


    }    
      //Move the particles in my group 1 dt in time.
    void GronbechJensen::forwardTime(){
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

	GronbechJensen_ns::integrateGPU<1><<<Nblocks, Nthreads, 0, stream>>>(pos.raw(),
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
	GronbechJensen_ns::integrateGPU<2><<<Nblocks, Nthreads, 0 , stream>>>(pos.raw(),
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

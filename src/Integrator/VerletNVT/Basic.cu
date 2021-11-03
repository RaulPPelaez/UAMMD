/*Raul P. Pelaez 2017-2021. Basic Verlet NVT Integrator module.
  See VerletNVT.cuh for more info
*/

#include"../VerletNVT.cuh"
#include"third_party/saruprng.cuh"
namespace uammd{
  namespace VerletNVT{
    namespace Basic_ns{
      //Fill the initial velocities of the particles in my group with a gaussian distribution according with my temperature.
      __global__ void initialVelocities(real3* vel, const real* mass, real defaultMass,
					ParticleGroup::IndexIterator indexIterator, //global index of particles in my group
					real vamp, bool is2D, int N, uint seed){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=N) return;
	Saru rng(id, seed);
	int i = indexIterator[id];
	real mass_i = 1.0;//mass?mass[i]:defaultMass;
	double3 noisei = make_double3(rng.gd(0, vamp/mass_i), is2D?0.0:rng.gd(0, vamp/mass_i).x);
	int index = indexIterator[i];
	vel[index].x = noisei.x;
	vel[index].y = noisei.y;
	vel[index].z = noisei.z;
      }
    }

    Basic::Basic(shared_ptr<ParticleGroup> pg,
		 Basic::Parameters par,
		 std::string name):
      Integrator(pg, name),
      dt(par.dt), temperature(par.temperature), friction(par.friction), is2D(par.is2D),
      steps(0){
      sys->rng().next32();
      sys->rng().next32();
      seed = sys->rng().next32();
      sys->log<System::MESSAGE>("[%s] Temperature: %f", name.c_str(), temperature);
      sys->log<System::MESSAGE>("[%s] Time step: %f", name.c_str(), dt);
      sys->log<System::MESSAGE>("[%s] Friction: %f", name.c_str(), friction);
      if(is2D){
	sys->log<System::MESSAGE>("[%s] Working in 2D mode.", name.c_str());
      }
      this->noiseAmplitude = sqrt(2*dt*friction*temperature);
      bool useDefaultMass = not pd->isMassAllocated();
      this->defaultMass = par.mass;
      if(useDefaultMass and this->defaultMass < 0 ){
	this->defaultMass = 1.0;
      }
      if(par.initVelocities)
	initVelocities();
      cudaStreamCreate(&stream);
    }

    void Basic::initVelocities(){
      int numberParticles = pg->getNumberParticles();
      if(pd->isVelAllocated()){
	sys->log<System::WARNING>("[%s] Velocity will be overwritten to ensure temperature conservation!", name.c_str());
      }
      auto vel_handle = pd->getVel(access::location::gpu, access::mode::write);
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      real velAmplitude = sqrt(3.0*temperature);
      auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read).raw();
      if(defaultMass > 0){
	mass = nullptr;
      }
      const int Nthreads = 128;
      const int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      Basic_ns::initialVelocities<<<Nblocks, Nthreads>>>(vel_handle.raw(),
							 mass, defaultMass,
							 groupIterator,
							 velAmplitude, is2D, numberParticles,
							 sys->rng().next32());
    }

    Basic::~Basic(){
      cudaStreamDestroy(stream);
    }

    namespace Basic_ns{

      //Integrate the movement 0.5 dt and reset the forces in the first step
      //velocity is updated half step using current forces
      //If step==1 additionally position is updated and force set to 0
      template<int step>
      __global__ void integrateGPU(real4* pos,
				   real3* vel,
				   real4* force,
				   const real* mass,
				   real defaultMass,
				   ParticleGroup::IndexIterator indexIterator,
				   int N,
				   real dt, real friction, bool is2D,
				   real noiseAmplitude,
				   uint stepNum, uint seed){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=N) return;
	//Index of current particle in group
	const int i = indexIterator[id];
	const real invMass = real(1.0)/(defaultMass>0?defaultMass:mass[i]);
	Saru rng(id+N*(step-1), stepNum, seed);
	noiseAmplitude *= sqrtf(0.5*invMass);
	const real3 noisei = make_real3(rng.gf(0, noiseAmplitude), rng.gf(0, noiseAmplitude).x);
	vel[i] += (make_real3(force[i])*invMass - friction*vel[i])*(dt*real(0.5)) + noisei;
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

    void Basic::resetForces(){
      int numberParticles = pg->getNumberParticles();
      auto force = pd->getForce(access::location::gpu, access::mode::write);
      auto force_group = pg->getPropertyIterator(force);
      thrust::fill(thrust::cuda::par.on(stream), force_group, force_group + numberParticles, real4());
    }

    template<int step>
    void Basic::callIntegrate(){
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
      int numberParticles = pg->getNumberParticles();
      int Nthreads = 128;
      int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      Basic_ns::integrateGPU<step><<<Nblocks, Nthreads, 0, stream>>>(pos.raw(),
								     vel.raw(),
								     force.raw(),
								     mass,
								     defaultMass,
								     groupIterator,
								     numberParticles, dt, friction, is2D,
								     noiseAmplitude,
								     steps, seed);
    }

    //Move the particles in my group 1 dt in time.
    void Basic::forwardTime(){
      for(auto updatable: updatables) updatable->updateSimulationTime(steps*dt);
      steps++;
      sys->log<System::DEBUG1>("[%s] Performing integration step %d", name.c_str(), steps);
      //First simulation step is special
      if(steps==1){
	resetForces();
	for(auto updatable: updatables){
	  updatable->updateTimeStep(dt);
	  updatable->updateTemperature(temperature);
	}
	for(auto forceComp: interactors) forceComp->sum({.force =true, .energy = false, .virial = false}, 0);
	cudaDeviceSynchronize();
      }
      //First integration step and force reset
      callIntegrate<1>();
      //Compute all the forces
      for(auto forceComp: interactors) forceComp->sum({.force =true, .energy = false, .virial = false}, stream);
      //Second integration step
      callIntegrate<2>();
    }

    namespace verletnvt_ns{
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

    real Basic::sumKineticEnergy(){
      int numberParticles = pg->getNumberParticles();
      auto vel_gr = pg->getPropertyIterator(pd->getVel(access::location::gpu, access::mode::read));
      auto energy_gr = pg->getPropertyIterator(pd->getEnergy(access::location::gpu, access::mode::readwrite));
      int Nthreads = 128;
      int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      if(defaultMass > 0){
	auto mass_iterator = thrust::make_constant_iterator(defaultMass);
	verletnvt_ns::sumKineticEnergy<<<Nblocks, Nthreads>>>(vel_gr, energy_gr, mass_iterator, numberParticles);
      }
      else{
	auto mass = pd->getMassIfAllocated(access::gpu, access::read);
	auto mass_iterator = pg->getPropertyIterator(mass);
	verletnvt_ns::sumKineticEnergy<<<Nblocks, Nthreads>>>(vel_gr, energy_gr, mass_iterator, numberParticles);
      }

      return 0.0;
    }
  }
}

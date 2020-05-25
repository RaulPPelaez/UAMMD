/*Raul P. Pelaez 2017-2020. Brownian Dynamics Integrator definitions
 */
#include"BrownianDynamics.cuh"
#include"third_party/saruprng.cuh"
#include"utils/debugTools.h"
namespace uammd{
  namespace BD{

    EulerMaruyama::EulerMaruyama(shared_ptr<ParticleData> pd,
				 shared_ptr<ParticleGroup> pg,
				 shared_ptr<System> sys,
				 Parameters par):
      Integrator(pd, pg, sys, "[BD::EulerMaruyama]"),
      Kx(make_real3(0)),
      Ky(make_real3(0)),
      Kz(make_real3(0)),
      temperature(par.temperature),
      dt(par.dt),
      is2D(par.is2D),
      steps(0){
      sys->rng().next32();
      sys->rng().next32();
      seed = sys->rng().next32();
      sys->log<System::MESSAGE>("[BD::EulerMaruyama] Initialized");
      int numberParticles = pg->getNumberParticles();
      this->selfMobility = 1.0/(6.0*M_PI*par.viscosity);
      if(par.hydrodynamicRadius != real(-1.0)){
	this->selfMobility /= par.hydrodynamicRadius;
	this->hydrodynamicRadius = par.hydrodynamicRadius;
	if(pd->isRadiusAllocated()){
	  sys->log<System::WARNING>("[BD::EulerMaruyama] Assuming all particles have hydrodynamic radius %f",
				    par.hydrodynamicRadius);
	}
	else{
	  sys->log<System::MESSAGE>("[BD::EulerMaruyama] Hydrodynamic radius: %f", par.hydrodynamicRadius);
	}
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Self Mobility: %f", selfMobility);
      }
      else if(pd->isRadiusAllocated()){
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Hydrodynamic radius: particleRadius");
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Self Mobility: %f/particleRadius", selfMobility);
      }
      else{
	this->hydrodynamicRadius = real(1.0);
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Hydrodynamic radius: %f", hydrodynamicRadius);
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Self Mobility: %f", selfMobility);
      }
      sys->log<System::MESSAGE>("[BD::EulerMaruyama] Temperature: %f", temperature);
      sys->log<System::MESSAGE>("[BD::EulerMaruyama] dt: %f", dt);
      this->sqrt2MTdt = sqrt(2.0*selfMobility*temperature*dt);
      if(par.K.size()==3){
	Kx = par.K[0];
	Ky = par.K[1];
	Kz = par.K[2];
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Shear Matrix: [ %f %f %f; %f %f %f; %f %f %f ]",
				  Kx.x, Kx.y, Kx.z,
				  Ky.x, Ky.y, Ky.z,
				  Kz.x, Kz.y, Kz.z);
      }
      if(is2D){
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Starting in 2D mode");
      }
      CudaSafeCall(cudaStreamCreate(&st));
    }

    EulerMaruyama::~EulerMaruyama(){
      sys->log<System::MESSAGE>("[BD::EulerMaruyama] Destroyed");
      cudaStreamDestroy(st);
    }

    namespace EulerMaruyama_ns{

      __global__ void integrateGPU(real4* pos,
				   ParticleGroup::IndexIterator indexIterator,
				   const real4* force,
				   real3 Kx, real3 Ky, real3 Kz,
				   real selfMobility,
				   const real* radius,
				   real dt,
				   bool is2D,
				   real sqrt2MTdt,
				   int N,
				   uint stepNum, uint seed){
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	int i = indexIterator[id];
	real3 p = make_real3(pos[i]);
	real3 f = make_real3(force[i]);
	real3 KR = make_real3(dot(Kx, p), dot(Ky, p), dot(Kz, p));
	real invRadius = real(1.0);
	if(radius){
	  invRadius = real(1.0)/radius[i];
	}
	// X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2·T·dt)·dW·B
	p += dt*( KR + selfMobility*invRadius*f);
	if(sqrt2MTdt > real(0.0)){ //When temperature > 0
	  Saru rng(id, stepNum, seed);
	  real sqrtInvRadius = real(1.0);
	  if(radius) sqrtInvRadius = sqrtf(invRadius);
	  const real noiseAmplitude = sqrt2MTdt*sqrtInvRadius;
	  real3 dW = make_real3(rng.gf(0, noiseAmplitude), 0);
	  if(!is2D)   dW.z = rng.gf(0, noiseAmplitude).x;
	  p += dW;
	}
	pos[i].x = p.x;
	pos[i].y = p.y;
	if(!is2D)
	  pos[i].z = p.z;
      }

    }

    void EulerMaruyama::forwardTime(){
      steps++;
      sys->log<System::DEBUG1>("[BD::EulerMaruyama] Performing integration step %d", steps);
      updateInteractors();
      resetForces();
      for(auto forceComp: interactors){
	forceComp->sumForce(st);
      }
      updatePositions();
    }
    void EulerMaruyama::updateInteractors(){
      for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*dt);
      if(steps==1){
	for(auto forceComp: interactors){
	  forceComp->updateTimeStep(dt);
	  forceComp->updateTemperature(temperature);
	}
      }
    }
    void EulerMaruyama::resetForces(){
      int numberParticles = pg->getNumberParticles();
      auto force = pd->getForce(access::location::gpu, access::mode::write);
      auto forceGroup = pg->getPropertyIterator(force);
      thrust::fill(thrust::cuda::par.on(st), forceGroup, forceGroup + numberParticles, real4());
    }

    real* EulerMaruyama::getParticleRadiusIfAvailable(){
      real* d_radius = nullptr;
      if(hydrodynamicRadius == real(-1.0) && pd->isRadiusAllocated()){
	auto radius = pd->getRadius(access::location::gpu, access::mode::read);
	d_radius = radius.raw();
	sys->log<System::DEBUG3>("[BD::EulerMaruyama] Using particle radius.");
      }
      return d_radius;
    }

    void EulerMaruyama::updatePositions(){
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      uint Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      uint Nblocks = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      real * d_radius = getParticleRadiusIfAvailable();
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      EulerMaruyama_ns::integrateGPU<<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
								   groupIterator,
								   force.raw(),
								   Kx, Ky, Kz,
								   selfMobility,
								   d_radius,
								   dt,
								   is2D,
								   sqrt2MTdt,
								   numberParticles,
								   steps, seed);

    }






    MidPoint::MidPoint(shared_ptr<ParticleData> pd,
		       shared_ptr<ParticleGroup> pg,
		       shared_ptr<System> sys,
		       Parameters par):
      Integrator(pd, pg, sys, "[BD::MidPoint]"),
      Kx(make_real3(0)),
      Ky(make_real3(0)),
      Kz(make_real3(0)),
      temperature(par.temperature),
      dt(par.dt),
      is2D(par.is2D),
      steps(0){
      sys->rng().next32();
      sys->rng().next32();
      seed = sys->rng().next32();
      sys->log<System::MESSAGE>("[BD::MidPoint] Initialized");
      int numberParticles = pg->getNumberParticles();
      this->selfMobility = 1.0/(6.0*M_PI*par.viscosity);
      if(par.hydrodynamicRadius != real(-1.0)){
	this->selfMobility /= par.hydrodynamicRadius;
	this->hydrodynamicRadius = par.hydrodynamicRadius;
	if(pd->isRadiusAllocated()){
	  sys->log<System::WARNING>("[BD::MidPoint] Assuming all particles have hydrodynamic radius %f",
				    par.hydrodynamicRadius);
	}
	else{
	  sys->log<System::MESSAGE>("[BD::MidPoint] Hydrodynamic radius: %f", par.hydrodynamicRadius);
	}
	sys->log<System::MESSAGE>("[BD::MidPoint] Self Mobility: %f", selfMobility);
      }
      else if(pd->isRadiusAllocated()){
	sys->log<System::MESSAGE>("[BD::MidPoint] Hydrodynamic radius: particleRadius");
	sys->log<System::MESSAGE>("[BD::MidPoint] Self Mobility: %f/particleRadius", selfMobility);
      }
      else{
	this->hydrodynamicRadius = real(1.0);
	sys->log<System::MESSAGE>("[BD::MidPoint] Hydrodynamic radius: %f", hydrodynamicRadius);
	sys->log<System::MESSAGE>("[BD::MidPoint] Self Mobility: %f", selfMobility);
      }
      sys->log<System::MESSAGE>("[BD::MidPoint] Temperature: %f", temperature);
      sys->log<System::MESSAGE>("[BD::MidPoint] dt: %f", dt);
      if(par.K.size()==3){
	Kx = par.K[0];
	Ky = par.K[1];
	Kz = par.K[2];
	sys->log<System::MESSAGE>("[BD::MidPoint] Shear Matrix: [ %f %f %f; %f %f %f; %f %f %f ]",
				  Kx.x, Kx.y, Kx.z,
				  Ky.x, Ky.y, Ky.z,
				  Kz.x, Kz.y, Kz.z);
      }
      if(is2D){
	sys->log<System::MESSAGE>("[BD::MidPoint] Starting in 2D mode");
      }
      this->sqrtDdt = sqrt(selfMobility*temperature*dt);
      CudaSafeCall(cudaStreamCreate(&st));
    }

    MidPoint::~MidPoint(){
      sys->log<System::MESSAGE>("[BD::MidPoint] Destroyed");
      cudaStreamDestroy(st);
    }

    namespace MidPoint_ns{

      template<int step>
      __global__ void integrateGPU(real4* pos,
				   real3* initialPositions,
				   ParticleGroup::IndexIterator indexIterator,
				   const real4* force,
				   real3 Kx, real3 Ky, real3 Kz,
				   real selfMobility,
				   const real* radius,
				   real dt,
				   bool is2D,
				   real sqrt2MTdt,
				   int N,
				   uint stepNum, uint seed){
        const auto id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	const int i = indexIterator[id];
	real3 p;
	if(step==0){
	  p = make_real3(pos[i]);
	  initialPositions[id] = p;
	}
	else if(step==1){
	  p = initialPositions[id];
	}
	real invRadius = real(1.0);
	if(radius){
	  invRadius = real(1.0)/radius[i];
	}
	real3 KR = make_real3(dot(Kx, p), dot(Ky, p), dot(Kz, p));
	real3 f = make_real3(force[i]);
	if(step == 0){
	  f*= real(0.5);
	  KR*=real(0.5);
	}
	p += dt*( KR + selfMobility*invRadius*f);
	if(sqrt2MTdt > real(0.0)){ //When temperature > 0
	  real sqrtInvRadius = real(1.0);
	  if(radius){
	    sqrtInvRadius = sqrtf(invRadius);
	  }
	  const real noiseAmplitude = sqrt2MTdt*sqrtInvRadius;
	  Saru rng(id, stepNum, seed);
	  real3 dW = make_real3(rng.gf(0, noiseAmplitude), 0);
	  if(!is2D)   dW.z = rng.gf(0, noiseAmplitude).x;
	  p += dW;
	  if(step==1){
	    dW = make_real3(rng.gf(0, noiseAmplitude), 0);
	    if(!is2D)   dW.z = rng.gf(0, noiseAmplitude).x;
	    p += dW;
	  }
	}
	pos[i].x = p.x;
	pos[i].y = p.y;
	if(!is2D)
	  pos[i].z = p.z;
      }
    }

    void MidPoint::forwardTime(){
      steps++;
      sys->log<System::DEBUG1>("[BD::MidPoint] Performing integration step %d", steps);
      updateInteractors();
      computeCurrentForces();
      updatePositionsFirstStep();
      computeCurrentForces();
      updatePositionsSecondStep();
      CudaCheckError();
    }

    void MidPoint::computeCurrentForces(){
      resetForces();
      for(auto forceComp: interactors){
	forceComp->sumForce(st);
      }
      CudaCheckError();
    }

    void MidPoint::updateInteractors(){
      for(auto forceComp: interactors){
	forceComp->updateSimulationTime(steps*dt);
      }
      if(steps==1){
	for(auto forceComp: interactors){
	  forceComp->updateTimeStep(dt);
	  forceComp->updateTemperature(temperature);
	}
      }
      CudaCheckError();
    }

    void MidPoint::resetForces(){
      int numberParticles = pg->getNumberParticles();
      auto force = pd->getForce(access::location::gpu, access::mode::write);
      auto forceGroup = pg->getPropertyIterator(force);
      thrust::fill(thrust::cuda::par.on(st), forceGroup, forceGroup + numberParticles, real4());
      CudaCheckError();
    }

    real* MidPoint::getParticleRadiusIfAvailable(){
      real* d_radius = nullptr;
      if(hydrodynamicRadius == real(-1.0) and pd->isRadiusAllocated()){
	auto radius = pd->getRadius(access::location::gpu, access::mode::read);
	d_radius = radius.raw();
	sys->log<System::DEBUG3>("[BD::MidPoint] Using particle radius.");
      }
      return d_radius;
    }

    void MidPoint::updatePositionsFirstStep(){
      updatePositions<0>();
    }

    void MidPoint::updatePositionsSecondStep(){
      updatePositions<1>();
    }

    template<int step>
    void MidPoint::updatePositions(){
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      uint Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      uint Nblocks = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      real * d_radius = getParticleRadiusIfAvailable();
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      initialPositions.resize(numberParticles);
      auto d_initialPositions = thrust::raw_pointer_cast(initialPositions.data());
      MidPoint_ns::integrateGPU<step><<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
								    d_initialPositions,
								    groupIterator,
								    force.raw(),
								    Kx, Ky, Kz,
								    selfMobility,
								    d_radius,
								    dt,
								    is2D,
								    sqrtDdt,
								    numberParticles,
								    steps, seed);
      CudaCheckError();
    }



  }
}

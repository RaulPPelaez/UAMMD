/*Raul P. Pelaez 2017-2020. Brownian Dynamics Integrator definitions
 */
#include"BrownianDynamics.cuh"
#include"third_party/saruprng.cuh"
#include"utils/debugTools.h"
namespace uammd{
  namespace BD{

    BaseBrownianIntegrator::BaseBrownianIntegrator(shared_ptr<ParticleData> pd,
						   shared_ptr<ParticleGroup> pg,
						   shared_ptr<System> sys,
						   Parameters par):
      Integrator(pd, pg, sys, "BD::BaseBrownianIntegrator"),
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
      sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Initialized");
      int numberParticles = pg->getNumberParticles();
      this->selfMobility = 1.0/(6.0*M_PI*par.viscosity);
      if(par.hydrodynamicRadius != real(-1.0)){
	this->selfMobility /= par.hydrodynamicRadius;
	this->hydrodynamicRadius = par.hydrodynamicRadius;
	if(pd->isRadiusAllocated()){
	  sys->log<System::WARNING>("[BD::BaseBrownianIntegrator] Assuming all particles have hydrodynamic radius %f",
				    par.hydrodynamicRadius);
	}
	else{
	  sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Hydrodynamic radius: %f", par.hydrodynamicRadius);
	}
	sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Self Mobility: %f", selfMobility);
      }
      else if(pd->isRadiusAllocated()){
	sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Hydrodynamic radius: particleRadius");
	sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Self Mobility: %f/particleRadius", selfMobility);
      }
      else{
	this->hydrodynamicRadius = real(1.0);
	sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Hydrodynamic radius: %f", hydrodynamicRadius);
	sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Self Mobility: %f", selfMobility);
      }
      sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Temperature: %f", temperature);
      sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] dt: %f", dt);
      if(par.K.size()==3){
	int numberNonZero = std::count_if(par.K.begin(), par.K.end(), [](real3 k){return k.x!=0 or k.y !=0 or k.z!=0;});
	if(numberNonZero>0){
	  Kx = par.K[0];
	  Ky = par.K[1];
	  Kz = par.K[2];
	  sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Shear Matrix: [ %f %f %f; %f %f %f; %f %f %f ]",
				    Kx.x, Kx.y, Kx.z,
				    Ky.x, Ky.y, Ky.z,
				    Kz.x, Kz.y, Kz.z);
	}
      }
      if(is2D){
	sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Starting in 2D mode");
      }
      CudaSafeCall(cudaStreamCreate(&st));
    }

    BaseBrownianIntegrator::~BaseBrownianIntegrator(){
      sys->log<System::MESSAGE>("[BD::BaseBrownianIntegrator] Destroyed");
      cudaStreamDestroy(st);
    }

    void BaseBrownianIntegrator::updateInteractors(){
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

    void BaseBrownianIntegrator::resetForces(){
      int numberParticles = pg->getNumberParticles();
      auto force = pd->getForce(access::location::gpu, access::mode::write);
      auto forceGroup = pg->getPropertyIterator(force);
      thrust::fill(thrust::cuda::par.on(st), forceGroup, forceGroup + numberParticles, real4());
      CudaCheckError();
    }

    void BaseBrownianIntegrator::computeCurrentForces(){
      resetForces();
      for(auto forceComp: interactors){
	forceComp->sumForce(st);
      }
      CudaCheckError();
    }

    real* BaseBrownianIntegrator::getParticleRadiusIfAvailable(){
      real* d_radius = nullptr;
      if(hydrodynamicRadius == real(-1.0) && pd->isRadiusAllocated()){
	auto radius = pd->getRadius(access::location::gpu, access::mode::read);
	d_radius = radius.raw();
	sys->log<System::DEBUG3>("[BD::BaseBrownianIntegrator] Using particle radius.");
      }
      return d_radius;
    }


    namespace EulerMaruyama_ns{

      __global__ void integrateGPU(real4* pos,
				   ParticleGroup::IndexIterator indexIterator,
				   const real4* force,
				   real3 Kx, real3 Ky, real3 Kz,
				   real selfMobility,
				   real* radius,
				   real dt,
				   bool is2D,
				   real temperature,
				   int N,
				   uint stepNum, uint seed){
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	int i = indexIterator[id];
	real3 R = make_real3(pos[i]);
	real3 F = make_real3(force[i]);
	real3 KR = make_real3(dot(Kx, R), dot(Ky, R), dot(Kz, R));
	real M = selfMobility*(radius?(real(1.0)/radius[i]):real(1.0));
	R += dt*( KR + M*F );
	if(temperature > 0){
	  Saru rng(i, stepNum, seed);
	  real B = sqrt(real(2.0)*temperature*M*dt);
	  real3 dW = make_real3(rng.gf(0, B), rng.gf(0, B).x);
	  R += dW;
	}
	pos[i].x = R.x;
	pos[i].y = R.y;
	if(!is2D)
	  pos[i].z = R.z;
      }

    }

    void EulerMaruyama::forwardTime(){
      steps++;
      sys->log<System::DEBUG1>("[BD::EulerMaruyama] Performing integration step %d", steps);
      updateInteractors();
      computeCurrentForces();
      updatePositions();
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
								   temperature,
								   numberParticles,
								   steps, seed);

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

    void MidPoint::updatePositionsFirstStep(){
      updatePositions<0>();
    }

    void MidPoint::updatePositionsSecondStep(){
      updatePositions<1>();
    }

    namespace MidPoint_ns{
      template<int step>
      __global__ void integrateGPU(real4* pos,
				   real4* initialPositions,
				   ParticleGroup::IndexIterator indexIterator,
				   const real4* force,
				   real3 Kx, real3 Ky, real3 Kz,
				   real selfMobility,
				   const real* radius,
				   real dt,
				   bool is2D,
				   real temperature,
				   int N,
				   uint stepNum, uint seed){
        const auto id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	const int i = indexIterator[id];
	real3 p;
	if(step==0){
	  p = make_real3(pos[i]);
	  initialPositions[id] = pos[i];
	}
	else if(step==1){
	  p = make_real3(initialPositions[id]);
	}
	const real M = selfMobility*(radius?(real(1.0)/radius[i]):real(1.0));
        real3 KR = make_real3(dot(Kx, p), dot(Ky, p), dot(Kz, p));
	real3 f = make_real3(force[i]);
	if(step == 0){
	  f *= real(0.5);
	  KR *= real(0.5);
	}
	p += dt*(KR + M*f);
	if(temperature > real(0.0)){
	  const real B = sqrt(temperature*M*dt);
	  Saru rng(id, stepNum, seed);
	  real3 dW = make_real3(rng.gf(0, B), rng.gf(0, B).x);
	  p += dW;
	  if(step==1){
	    dW = make_real3(rng.gf(0, B), rng.gf(0, B).x);
	    p += dW;
	  }
	}
	pos[i].x = p.x;
	pos[i].y = p.y;
	if(!is2D)
	  pos[i].z = p.z;
      }
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
								    temperature,
								    numberParticles,
								    steps, seed);
      CudaCheckError();
    }


    void AdamsBashforth::forwardTime(){
      steps++;
      sys->log<System::DEBUG1>("[BD::AdamsBashforth] Performing integration step %d", steps);
      if(steps==1){
	updateInteractors();
	computeCurrentForces();
      }
      storeCurrentForces();
      updateInteractors();
      computeCurrentForces();
      updatePositions();
    }

    void AdamsBashforth::storeCurrentForces(){
      int numberParticles = pg->getNumberParticles();
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      auto forceGroup = pg->getPropertyIterator(force);
      previousForces.resize(numberParticles);
      thrust::copy(thrust::cuda::par.on(st), forceGroup, forceGroup + numberParticles, previousForces.begin());
      CudaCheckError();
    }

    namespace AdamsBashforth_ns{
      __global__ void integrateGPU(real4* pos,
				   real4* previousForces,
				   ParticleGroup::IndexIterator indexIterator,
				   const real4* force,
				   real3 Kx, real3 Ky, real3 Kz,
				   real selfMobility,
				   const real* radius,
				   real dt,
				   bool is2D,
				   real temperature,
				   int N,
				   uint stepNum, uint seed){
        const auto id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	const int i = indexIterator[id];
	real3 p = make_real3(pos[i]);
	const real M = selfMobility*(radius?(real(1.0)/radius[i]):real(1.0));
        real3 KR = make_real3(dot(Kx, p), dot(Ky, p), dot(Kz, p));
	real3 fn = make_real3(force[i]);
	real3 fprev = make_real3(previousForces[id]);
	p += dt*(KR + M*(real(1.5)*fn - real(0.5)*fprev));
	if(temperature > real(0.0)){
	  const real B = sqrt(real(2.0)*temperature*M*dt);
	  Saru rng(id, stepNum, seed);
	  real3 dW = make_real3(rng.gf(0, B), rng.gf(0, B).x);
	  p += dW;
	}
	pos[i].x = p.x;
	pos[i].y = p.y;
	if(!is2D)
	  pos[i].z = p.z;
      }
    }

    void AdamsBashforth::updatePositions(){
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      uint Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      uint Nblocks = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      real * d_radius = getParticleRadiusIfAvailable();
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      previousForces.resize(numberParticles);
      auto d_previousForces = thrust::raw_pointer_cast(previousForces.data());
      AdamsBashforth_ns::integrateGPU<<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
								    d_previousForces,
								    groupIterator,
								    force.raw(),
								    Kx, Ky, Kz,
								    selfMobility,
								    d_radius,
								    dt,
								    is2D,
								    temperature,
								    numberParticles,
								    steps, seed);
      CudaCheckError();
    }


    namespace Leimkuhler_ns{

      __device__ real3 genNoise(int i, uint stepNum, uint seed){
	Saru rng(i, stepNum, seed);
	return make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
      }

      __global__ void integrateGPU(real4* pos,
				   ParticleGroup::IndexIterator indexIterator,
				   const int* originalIndex,
				   const real4* force,
				   real3 Kx, real3 Ky, real3 Kz,
				   real selfMobility,
				   real* radius,
				   real dt,
				   bool is2D,
				   real temperature,
				   int N,
				   uint stepNum, uint seed){
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	int i = indexIterator[id];
	real3 R = make_real3(pos[i]);
	real3 F = make_real3(force[i]);
	real3 KR = make_real3(dot(Kx, R), dot(Ky, R), dot(Kz, R));
	real M = selfMobility*(radius?(real(1.0)/radius[i]):real(1.0));
	R += dt*( KR + M*F );
	if(temperature > 0){
	  int ori = originalIndex[i];
	  real B = sqrt(real(0.5)*temperature*M*dt);
	  real3 dW = genNoise(ori, stepNum, seed) + genNoise(ori, stepNum-1, seed);
	  R += B*dW;
	}
	pos[i].x = R.x;
	pos[i].y = R.y;
	if(!is2D)
	  pos[i].z = R.z;
      }

    }

    void Leimkuhler::forwardTime(){
      steps++;
      sys->log<System::DEBUG1>("[BD::Leimkuhler] Performing integration step %d", steps);
      updateInteractors();
      computeCurrentForces();
      updatePositions();
    }

    void Leimkuhler::updatePositions(){
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      uint Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      uint Nblocks = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      real * d_radius = getParticleRadiusIfAvailable();
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      auto originalIndex = pd->getIdOrderedIndices(access::location::gpu);
      Leimkuhler_ns::integrateGPU<<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
								groupIterator,
								originalIndex,
								force.raw(),
								Kx, Ky, Kz,
								selfMobility,
								d_radius,
								dt,
								is2D,
								temperature,
								numberParticles,
								steps, seed);

    }



  }
}

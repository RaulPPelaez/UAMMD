// Brownian Dynamics with rotation Integrator definitions
namespace uammd{
  namespace extensions{
    namespace BDR{
      
      BrownianRotation::BrownianRotation(shared_ptr<ParticleData> pd,
					 shared_ptr<ParticleGroup> pg,
					 shared_ptr<System> sys,
					 BrownianRotation::Parameters par):
	Integrator(pd, pg, sys, "BrownianRotation"),
	Kx(make_real3(0)),
	Ky(make_real3(0)),
	Kz(make_real3(0)),
	temperature(par.temperature),
	dt(par.dt),
	steps(0){  
	sys->log<System::MESSAGE>("[BrownianRotation] Time step: %.3f", dt);
	sys->rng().next32();
	sys->rng().next32();
	seed = sys->rng().next32();
	this -> rotSelfMobility = 1.0/(8.0*M_PI*par.viscosity);
	this -> translSelfMobility = 1.0/(6.0*M_PI*par.viscosity);
	if(par.hydrodynamicRadius != real(-1.0)){
	  this -> rotSelfMobility /= pow(par.hydrodynamicRadius,3);
	  this -> translSelfMobility /= par.hydrodynamicRadius;
	  this -> hydrodynamicRadius = par.hydrodynamicRadius;
	} else {
	  this -> hydrodynamicRadius = real(1.0);
	}
      	sys->log<System::MESSAGE>("[BDR::BaseBrownianRotation] Hydrodynamic radius: %f", par.hydrodynamicRadius);
	sys->log<System::MESSAGE>("[BDR::BaseBrownianRotation] Translational Self Mobility: %f", translSelfMobility);
	sys->log<System::MESSAGE>("[BDR::BaseBrownianRotation] Rotational Self Mobility: %f", rotSelfMobility);
	sys->log<System::MESSAGE>("[BDR::BaseBrownianRotation] Temperature: %f", temperature);
	sys->log<System::MESSAGE>("[BDR::BaseBrownianRotation] dt: %f", dt);
	
	CudaSafeCall(cudaStreamCreate(&stream));
      }


      BrownianRotation::~BrownianRotation(){
	sys->log<System::MESSAGE>("[BrownianRotation] Destroyed");
	cudaStreamDestroy(stream);
      }

      void BrownianRotation::updateInteractors(){
	for(auto forceComp: interactors){
	  forceComp->updateSimulationTime(steps*dt);
	}
	if(steps==1){
	  for(auto forceComp: interactors){
	    forceComp->updateTimeStep(dt);
	  }
	}
	CudaCheckError();
      }
      
      void BrownianRotation::resetTorques(){
	int numberParticles = pg->getNumberParticles();
	auto torque = pd->getTorque(access::location::gpu, access::mode::write);
	auto torqueGroup = pg->getPropertyIterator(torque);
	thrust::fill(thrust::cuda::par.on(stream), torqueGroup, torqueGroup + numberParticles, real4());
        CudaCheckError();
      }
      
      void BrownianRotation::resetForces(){
	int numberParticles = pg->getNumberParticles();
	auto force = pd->getForce(access::location::gpu, access::mode::write);
	auto forceGroup = pg->getPropertyIterator(force);
	thrust::fill(thrust::cuda::par.on(stream), forceGroup, forceGroup + numberParticles, real4());
	CudaCheckError();
      }

      void BrownianRotation::computeCurrentForces(){
	resetForces();
	resetTorques();
	for(auto forceComp: interactors){
	  forceComp->sumForce(stream);
	}
	CudaCheckError();
      }
      
      namespace BrownianRotation_ns{

	__global__ void integrateGPU(real4*  pos,
				     real4* dir,
				     const real4* force,
				     const real4* torque,
				     real3 Kx, real3 Ky, real3 Kz,
				     ParticleGroup::IndexIterator indexIterator,
				     real temperature,
				     uint seed, int N,
				     real dt, int step,
				     real rotSelfMobility, real translSelfMobility){
	
	  int id = blockIdx.x*blockDim.x+threadIdx.x;
	  if(id>=N) return;
	  int i = indexIterator[id];
	  Saru rng(seed,i,step);
	  
	  // Translational movement
	  real Mt = translSelfMobility;
	  real3 R = make_real3(pos[i]);
	  real3 KR = make_real3(dot(Kx, R), dot(Ky, R), dot(Kz, R));
	  real3 F = make_real3(force[i]);
	  real stdt =sqrt(2*temperature*Mt*dt);
	  real3 dWt =make_real3(rng.gf(0, stdt), rng.gf(0, stdt).x);
	  R += dt*( KR + stdt*F ) + dWt;
	  pos[i].x = R.x;
	  pos[i].y = R.y;
	  pos[i].z = R.z;
	  
	  //Rotational movement
	  real Mr = rotSelfMobility;
	  real stdr = sqrt(2*Mr*temperature*dt);
	  real3 dWr = make_real3(rng.gf(0, stdr), rng.gf(0, stdr).x);
	  real3 tor = make_real3(torque[i]);
	  real3 dphi = Mr*dt*tor + dWr;
	  Quat newDir = dir[i];
	  dir[i] = make_real4(rotVec2Quaternion(dphi)*newDir);	
	}
      };
      
      void BrownianRotation::updatePositions(){
	int nParticles = pg->getNumberParticles();
	int numberParticles = pg->getNumberParticles();
	int BLOCKSIZE = 128;
	uint Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
	uint Nblocks = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
	auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	auto force = pd->getForce(access::location::gpu, access::mode::read);
	auto dir = pd->getDir(access::location::gpu, access::mode::readwrite);
	auto tor = pd->getTorque(access::location::gpu, access::mode::read);
	auto groupIterator = pg->getIndexIterator(access::location::gpu);
	
	BrownianRotation_ns::integrateGPU<<<Nblocks, Nthreads, 0, stream>>>(pos.begin(),
									    dir.begin(),
									    force.begin(),
									    tor.begin(),
									    Kx, Ky, Kz,
									    groupIterator,
									    temperature, seed,
									    nParticles, dt, steps,
									    rotSelfMobility,
									    translSelfMobility);
      }
      
      void BrownianRotation::forwardTime(){
	steps++;
	sys->log<System::DEBUG1>("[BrownianRotation] Performing integration step %d", steps);  
	updateInteractors();
	computeCurrentForces();
	updatePositions();
      }
    }
  }
}

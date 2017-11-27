/*Raul P. Pelaez 2017. Brownian Euler Maruyama Integrator definition

  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*Tdt)·dW·B
   Being:
     X - Positions
     M - Self Diffusion  coefficient -> 1/(6·pi·vis·radius)
     K - Shear matrix
     dW- Noise vector
     B - sqrt(M)
*/
#include"BrownianDynamics.cuh"

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

      sys->log<System::MESSAGE>("[BD::EulerMaruyama] Initialized");
      
      int numberParticles = pg->getNumberParticles();

      if(temperature>real(0.0)){
	noise.resize(3*numberParticles + ((3*numberParticles)%2));
	curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next());
	auto d_noise = thrust::raw_pointer_cast(noise.data());
	curandGenerateNormal(curng, d_noise, noise.size(),  0.0, 1.0);	
      }


      this->selfDiffusion = 1.0/(6.0*M_PI*par.viscosity);

      sys->log<System::MESSAGE>("[BD::EulerMaruyama] Temperature: %f", temperature);
      sys->log<System::MESSAGE>("[BD::EulerMaruyama] dt: %f", dt);

      
      if(par.hydrodynamicRadius != real(-1.0)){
	this->selfDiffusion /= par.hydrodynamicRadius;
	this->hydrodynamicRadius = par.hydrodynamicRadius;
	if(pd->isRadiusAllocated()){
	  sys->log<System::WARNING>("[BD::EulerMaruyama] Assuming all particles have hydrodynamic radius %f",
				    par.hydrodynamicRadius);
	}
	else{
	  sys->log<System::MESSAGE>("[BD::EulerMaruyama] Hydrodynamic radius: %f", par.hydrodynamicRadius);
	}
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Self Diffusion: %f", selfDiffusion);
      }
      else if(pd->isRadiusAllocated()){
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Hydrodynamic radius: particleRadius");
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Self Diffusion: %f/particleRadius",
				    selfDiffusion);      
      }
      else{
	//Default hydrodynamic radius when none is provided is 1
	this->hydrodynamicRadius = real(1.0);
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Hydrodynamic radius: %f", hydrodynamicRadius);
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Self Diffusion: %f", selfDiffusion);
      }      


      this->sqrt2MTdt = sqrt(2.0*selfDiffusion*temperature*dt);



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

      cudaStreamCreate(&noiseStream);
      cudaStreamCreate(&forceStream);

    }
    

    EulerMaruyama::~EulerMaruyama(){
      sys->log<System::MESSAGE>("[BD::EulerMaruyama] Destroyed");
      if(this->is2D)curandDestroyGenerator(curng);
      cudaStreamDestroy(noiseStream);
      cudaStreamDestroy(forceStream);		     
    }

    
    namespace EulerMaruyama_ns{
      /*Integrate the movement*/
      __global__ void integrateGPU(real4 __restrict__  *pos,
				   ParticleGroup::IndexIterator indexIterator,
				   const real4 __restrict__  *force,
				   const real3 __restrict__ *dW,
				   real3 Kx, real3 Ky, real3 Kz,
				   real selfDiffusion,
				   real * radius,
				   real dt,
				   bool is2D,
				   real sqrt2MTdt,
				   int N){
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;

	int i = indexIterator[id];
	/*Half step velocity*/	
	real3 p = make_real3(pos[i]);
	real3 f = make_real3(force[i]);

	real3 KR = make_real3(dot(Kx, p),
			      dot(Ky, p),
			      dot(Kz, p));

	real invRadius = real(1.0);
	if(radius) invRadius = real(1.0)/radius[i];
	// X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2·T·dt)·dW·B
	p += dt*( KR + selfDiffusion*invRadius*f);
	if(dW){ //When temperature > 0
	  real sqrtInvRadius = real(1.0);
	  if(radius) sqrtInvRadius = sqrtf(invRadius);
	  p += sqrt2MTdt*dW[i]*sqrtInvRadius;
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

      for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*dt);

      if(steps==1){
	for(auto forceComp: interactors){
	  forceComp->updateTimeStep(dt);
	  forceComp->updateTemperature(temperature);	 
	}
      }
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      uint Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      uint Nblocks = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);

      real * d_radius = nullptr;
      if(hydrodynamicRadius == real(-1.0) && pd->isRadiusAllocated()){
	auto radius = pd->getRadius(access::location::gpu, access::mode::read);
	d_radius = radius.raw();
      }
      

      real3 * d_noise = nullptr;
      if(temperature > real(0.0)){
	curandSetStream(curng, noiseStream);
	noise.resize(3*numberParticles + ((3*numberParticles)%2));
	d_noise = (real3*)thrust::raw_pointer_cast(noise.data());
	curandGenerateNormal(curng, (real*)d_noise, noise.size(),  0.0, 1.0);	
      }

      
      /*Compute new forces*/
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      {
	auto force = pd->getForce(access::location::gpu, access::mode::write);      
	fillWithGPU<<<Nblocks, Nthreads>>>(force.raw(), groupIterator, make_real4(0), numberParticles);
      }
      for(auto forceComp: interactors) forceComp->sumForce(forceStream);
    
      

      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      EulerMaruyama_ns::integrateGPU<<<Nblocks, Nthreads>>>(pos.raw(),
							    groupIterator,
							    force.raw(),
							    d_noise,
							    Kx, Ky, Kz,
							    selfDiffusion,
							    d_radius,
							    dt,
							    is2D,
							    sqrt2MTdt,
							    numberParticles);

    }
    

  }
}
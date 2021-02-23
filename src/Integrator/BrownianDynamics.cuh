/*Raul P. Pelaez 2017-2020. Brownian Dynamics integrators

  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*Tdt)·dW·B
   Being:
     X - Positions
     M - Self Diffusion  coefficient -> 1/(6·pi·vis·radius)
     K - Shear matrix
     dW- Noise vector
     B - sqrt(M)


OPTIONS:

BD::Parameters par;
par.K -> a std:vector<real3> of three elements, encoding a 3x3 shear Matrix. zero(3,3) by default.
par.temperature -> System Temperature
par.viscosity -> System Viscosity
par.hydrodynamicRadius -> Particle radius (if all particles have the same radius). Set this variable if pd->radius has not been set or you want all particles to have the same diffusive radius and ignore pd->radius.
par.dt -> Time step size.
par.is2D -> Set to true if the system lives in 2D.

USAGE:
Use as any other Integrator.

  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N, sys);
  ...
//Set initial state
  ...
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  BD::Parameters par;
  par.temperature = std::stod(argv[7]); //For example
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = std::stod(argv[3]); //For example
  using Scheme = BD::EulerMaruyama;
//using Scheme = BD::MidPoint;
  auto bd = make_shared<Scheme>(pd, pg, sys, par);

See examples/BD.cu for an example

References:

[1] Temporal Integrators for Fluctuating Hydrodynamics. Delong et. al. (2013) Phys. Rev. E 87, 033302.
[2] Brownian dynamics of confined suspensions of active microrollers. Balboa et. al. (2017) J. Chem. Phys. 146; https://doi.org/10.1063/1.4979494
*/
#ifndef BROWNIANDYNAMICSINTEGRATOR_CUH
#define BROWNIANDYNAMICSINTEGRATOR_CUH
#include"global/defines.h"
#include"Integrator.cuh"

namespace uammd{
  namespace BD{
    struct Parameters{
      //The 3x3 shear matrix is encoded as an array of 3 real3
      std::vector<real3> K;
      real temperature = 0;
      real viscosity = 1;
      real hydrodynamicRadius = -1.0;
      real dt = 0;
      bool is2D = false;
    };


    class BaseBrownianIntegrator: public Integrator{
    public:
      using Parameters = BD::Parameters;

      BaseBrownianIntegrator(shared_ptr<ParticleData> pd,
		    shared_ptr<ParticleGroup> pg,
		    shared_ptr<System> sys,
		    Parameters par);

      BaseBrownianIntegrator(shared_ptr<ParticleData> pd,
			     shared_ptr<System> sys,
			     Parameters par):
	BaseBrownianIntegrator(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par){}

      ~BaseBrownianIntegrator();

      virtual void forwardTime() = 0;

      virtual real sumEnergy() override{
	//Sum 1.5*kT to each particle
	auto energy = pd->getEnergy(access::gpu, access::readwrite);
	auto energy_gr = pg->getPropertyIterator(energy);
	auto energy_per_particle = thrust::make_constant_iterator<real>(1.5*temperature);
	thrust::transform(thrust::cuda::par,
			  energy_gr, energy_gr + pg->getNumberParticles(),
			  energy_per_particle,
			  energy_gr,
			  thrust::plus<real>());
	return 0;
      }

    protected:
      real3 Kx, Ky, Kz; //shear matrix
      real selfMobility;
      real hydrodynamicRadius = real(-1.0);
      real temperature = real(0.0);
      real dt;
      bool is2D;
      cudaStream_t st;
      int steps;
      uint seed;

      void updateInteractors();
      void resetForces();
      void computeCurrentForces();
      real* getParticleRadiusIfAvailable();
    };


    class EulerMaruyama: public BaseBrownianIntegrator{
    public:
      EulerMaruyama(shared_ptr<ParticleData> pd,
		    shared_ptr<ParticleGroup> pg,
		    shared_ptr<System> sys,
		    Parameters par):BaseBrownianIntegrator(pd, pg, sys, par){
	sys->log<System::MESSAGE>("[BD::EulerMaruyama] Initialized");
      }

      EulerMaruyama(shared_ptr<ParticleData> pd,
		    shared_ptr<System> sys,
		    Parameters par):
	EulerMaruyama(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par){}

      void forwardTime() override;

    protected:
      void updatePositions();
    };

    //Implements the algorithm in [1]
    class MidPoint: public BaseBrownianIntegrator{
    public:
      MidPoint(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       Parameters par):BaseBrownianIntegrator(pd, pg, sys, par){
	sys->log<System::MESSAGE>("[BD::MidPoint] Initialized");
      }

      MidPoint(shared_ptr<ParticleData> pd,
	      shared_ptr<System> sys,
	      Parameters par):
	MidPoint(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par){}

      void forwardTime() override;

    protected:
      thrust::device_vector<real4> initialPositions;
      void updatePositionsFirstStep();
      void updatePositionsSecondStep();
      template<int step> void updatePositions();
    };

    //Implements the algorithm in eq. 45 of [2]
    class AdamsBashforth: public BaseBrownianIntegrator{
    public:
      AdamsBashforth(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       Parameters par):BaseBrownianIntegrator(pd, pg, sys, par){
	sys->log<System::MESSAGE>("[BD::AdamsBashforth] Initialized");
      }

      AdamsBashforth(shared_ptr<ParticleData> pd,
	      shared_ptr<System> sys,
	      Parameters par):
	AdamsBashforth(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par){}

      void forwardTime() override;

    private:
      thrust::device_vector<real4> previousForces;
      void storeCurrentForces();
      void updatePositions();
    };

    
    //Implements the algorithm in eq. 45 of [2]
    class Leimkuhler: public BaseBrownianIntegrator{
    public:
      Leimkuhler(shared_ptr<ParticleData> pd,
		     shared_ptr<ParticleGroup> pg,
		     shared_ptr<System> sys,
		     Parameters par):BaseBrownianIntegrator(pd, pg, sys, par){
	sys->log<System::MESSAGE>("[BD::Leimkuhler] Initialized");
      }

      Leimkuhler(shared_ptr<ParticleData> pd,
		     shared_ptr<System> sys,
		     Parameters par):
	Leimkuhler(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par){}

      void forwardTime() override;

    private:
      void updatePositions();
    };

    
  }
}

#include"BrownianDynamics.cu"
#endif

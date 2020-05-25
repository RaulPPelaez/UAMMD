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

    class EulerMaruyama: public Integrator{
    public:
      using Parameters = BD::Parameters;

      EulerMaruyama(shared_ptr<ParticleData> pd,
		    shared_ptr<ParticleGroup> pg,
		    shared_ptr<System> sys,
		    Parameters par);

      EulerMaruyama(shared_ptr<ParticleData> pd,
		    shared_ptr<System> sys,
		    Parameters par):
	EulerMaruyama(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par){}

      ~EulerMaruyama();

      void forwardTime() override;

    protected:
      real3 Kx, Ky, Kz; //shear matrix
      real selfMobility;
      real hydrodynamicRadius = real(-1.0);
      real temperature = real(0.0);
      real sqrt2MTdt;
      real dt;
      bool is2D;
      cudaStream_t st;
      int steps;
      uint seed;

      void updateInteractors();
      void resetForces();
      real* getParticleRadiusIfAvailable();
      void updatePositions();
    };

    //Implements the algorithm in [1]
    class MidPoint: public Integrator{
    public:
      using Parameters = BD::Parameters;

      MidPoint(shared_ptr<ParticleData> pd,
	      shared_ptr<ParticleGroup> pg,
	      shared_ptr<System> sys,
	      Parameters par);

      MidPoint(shared_ptr<ParticleData> pd,
	      shared_ptr<System> sys,
	      Parameters par):
	MidPoint(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par){}

      ~MidPoint();

      void forwardTime() override;

    protected:
      real3 Kx, Ky, Kz;
      real selfMobility;
      real hydrodynamicRadius = real(-1.0);
      real temperature = real(0.0);
      real sqrtDdt;
      real dt;
      bool is2D;
      cudaStream_t st;
      int steps;
      uint seed;
      thrust::device_vector<real3> initialPositions;
      void updateInteractors();
      void resetForces();
      void computeCurrentForces();
      real* getParticleRadiusIfAvailable();
      void updatePositionsFirstStep();
      void updatePositionsSecondStep();
      template<int step> void updatePositions();

    };

  }
}

#include"BrownianDynamics.cu"
#endif

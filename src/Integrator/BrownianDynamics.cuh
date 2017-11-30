/*Raul P. Pelaez 2017. Brownian Euler Maruyama

  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*Tdt)·dW·B
   Being:
     X - Positions
     M - Self Diffusion  coefficient -> 1/(6·pi·vis·radius)
     K - Shear matrix
     dW- Noise vector
     B - sqrt(M)




OPTIONS:

BD::EulerMaruyama::Parameters par;

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
Set initial state
  ...

  auto pg = make_shared<ParticleGroup>(pd, sys, "All");

  BD::EulerMaruyama::Parameters par;
  par.temperature = std::stod(argv[7]); //For example
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = std::stod(argv[3]); //For example

  auto bd = make_shared<BD::EulerMaruyama>(pd, pg, sys, par);

See exampleS/BD.cu for an example  

*/
#ifndef BROWNIANEULERMARUYAMAINTEGRATOR_CUH
#define BROWNIANEULERMARUYAMAINTEGRATOR_CUH
#include"global/defines.h"
#include"Integrator.cuh"
#include<curand.h>

#ifndef SINGLE_PRECISION
#define curandGenerateNormal curandGenerateNormalDouble
#endif

namespace uammd{
  namespace BD{    
    class EulerMaruyama: public Integrator{
    public:
      struct Parameters{
	//The 3x3 shear matrix is encoded as an array of 3 real3
	std::vector<real3> K;
	real temperature = 0;
	real viscosity = 1;
	real hydrodynamicRadius = -1.0;
	real dt = 0;
	bool is2D = false;
      };

      //Constructor, you have to provide D and K.
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
  
      thrust::device_vector<real> noise;
      curandGenerator_t curng;

      bool is2D;

      cudaStream_t noiseStream, forceStream;
      
      int steps;
    };
  }
}

#include"BrownianDynamics.cu"
#endif

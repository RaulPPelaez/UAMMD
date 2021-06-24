/* 
   Solves the Brownian rotation and translation of spherical particles.
   The translational part follows the Euler Maruyama scheme of BrownyanDynamics.cuh
   
   The rotational part is solved using quaternions and their properties to encode
   the orientations and the rotations of the particles. [1]
   
   The quaternion that encodes the orientation the particle i is a real4 contained in dir[i]
 
   Translational differential equation:
   X[t+dt] = dt(K·X[t]+Mt·F[t]) + sqrt(2*Tdt)·dW·Bt
   Being:
     X - Positions
     Mt - Translational Self Diffusion  coefficient -> 1/(6·pi·vis·radius)
     dW- Noise vector
     K - Shear matrix
     Bt - sqrt(Mt)
   
   Rotational differential equation:
   dphi = dt(Mr·Tor[t])+sqrt(2·T·dt)·dW·Br
   Q[t+dt] = dQ*Q[t+dt]
   Being:
      dphi - Rotation vector
      dQ - Quaternion encoding the rotation vector
      *  - Product between quaternions
      Q  - Quaternion encoding the orientation of the particles
      Mr - Rotational Self Diffusion coefficient -> 1/(8·pi·vis·radius^3)
      Br - sqrt(Mr)
   
   Given two quaternions q1(n1,v1) and q2(n2,v2) the product q3(n3,v3) = q1*q2 is defined as:
   q3 = q1*q2 = (n1·n2 - v1·v2, n1·v2 + n2·v1 + v1 x v2)

   Both differential equations are solved using the Euler-Maruyama scheme


OPTIONS:

BDR::Parameters par;
par.K -> a std:vector<real3> of three elements, encoding a 3x3 shear Matrix. zero(3,3) by default.
par.temperature -> System Temperature
par.viscosity -> System Viscosity
par.hydrodynamicRadius -> Particle radius (if all particles have the same radius). Set this variable if pd->radius has not been set or you want all particles to have the same diffusive radius and ignore pd->radius.
par.dt -> Time step size.
   

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

The orientations can be initialized with the function quaternion::initOrientations (int nParticles, uint seed, std::string option);
Being:
  nParticles -> The number of particles in the system
  seed -> A seed for generating random numbers (if no seed provided it takes std::time(NULL))
  type -> option for initialization:
         "Random" -> Random orientations uniformly distributed
	 "aligned" -> All the particles are aligned with the laboratory frame

References:
     [1] https://aip.scitation.org/doi/10.1063/1.4932062

   Contributors:
     - Raul P. Pelaez     -> Translational part
     - P. Palacios Alonso -> Rotational part
*/

#include "Integrator/Integrator.cuh"
#include <curand.h>
#include <thrust/device_vector.h>
#include"utils/quaternion.cuh"
namespace uammd{
  namespace extensions{
    namespace BDR{
      struct Parameters{
	std::vector<real3> K = std::vector<real3>(3,real3());
	real temperature = 0;
	real viscosity = 1.0;
	real hydrodynamicRadius = -1.0;
	real dt = 0.0;
      };
    
      class BrownianRotation: public Integrator{
      public:
	using Parameters = BDR::Parameters;
      
	BrownianRotation(shared_ptr<ParticleData> pd,
			 shared_ptr<ParticleGroup> pg,
			 shared_ptr<System> sys,
			 Parameters par);
	
	BrownianRotation(shared_ptr<ParticleData> pd,
			 shared_ptr<System> sys,
			 Parameters par):
	  BrownianRotation(pd, std::make_shared<ParticleGroup>(pd, sys),sys, par){}
      
	~BrownianRotation();
      
	virtual void forwardTime() override;

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
	real dt;
	real rotSelfMobility;
	real translSelfMobility;
	real hydrodynamicRadius;
	cudaStream_t stream;
	int steps;   
	uint seed;
	real temperature;
  
	void updateInteractors();
	void resetForces();
	void resetTorques();
	void computeCurrentForces();   
	void updatePositions();
	real* getParticleRadiusIfAvailable();
      };
    }
  }
}
#include"BrownianRotation.cu"

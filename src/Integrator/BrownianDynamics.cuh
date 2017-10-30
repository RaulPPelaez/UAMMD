/*Raul P. Pelaez 2017. Brownian Euler Maruyama Integrator derived class implementation

  An Integrator is intended to be a separated module that handles the update of positions given the forces

  Solves the following differential equation:
      X[t+dt] = dt(K·X[t]+M·F[t]) + sqrt(2*Tdt)·dW·B
   Being:
     X - Positions
     M - Self Diffusion  coefficient
     K - Shear matrix
     dW- Noise vector
     B - sqrt(M)
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
	real temperature;
	real viscosity = 1;
	real hydrodynamicRadius = 0.5;      
	real dt;
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

    private:
      real3 Kx, Ky, Kz; //shear matrix
      real selfDiffusion;
      real temperature;
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

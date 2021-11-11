/*Raul P. Pelaez 2017-2020. Positively Split Edwald Rotne-Prager-Yamakawa Brownian Dynamics with Hydrodynamic interactions.


  As this is a BDHI module. BDHI_PSE computes the terms M·F and B·dW in the differential equation:
            dR = K·R·dt + M·F·dt + sqrt(2Tdt)· B·dW

  The mobility, M, is computed according to the Rotne-Prager-Yamakawa (RPY) tensor.

  The computation uses periodic boundary conditions (PBC)
  and partitions the RPY tensor in two, positively defined contributions [1], so that:
      M = Mr + Mw
       Mr - A real space short range contribution.
       Mw - A wave space long range contribution.

  Such as:
     M·F = Mr·F + Mw·F
     B·dW = sqrt(Mr)·dWr + sqrt(Mw)·dWw
####################      Short Range     #########################


  Mr·F: The short range contribution of M·F is computed using a neighbour list (this is like a sparse matrix-vector product in which each element is computed on the fly), see PSE_ns::RPYNearTransverser.
        The RPY near part function (see Apendix 1 in [1]) is precomputed and stored in texture memory,
	see PSE_ns::RPYPSE_nearTextures.

  sqrt(Mr)·dW: The near part stochastic contribution is computed using the Lanczos algorithm (see misc/LanczosAlgorithm.cuh), the function that computes M·v is provided via a functor called PSE_ns::Dotctor, the logic of M·v itself is the same as in M·F (see PSE_ns::RPYNearTransverser) and is computed with the same neighbour list.

###################        Far range     ###########################



  Mw·F:  Mw·F = σ·St·FFTi·B·FFTf·S · F. The long range wave space part.
         -σ: The volume of a grid cell
	 -S: An operator that spreads each element of a vector to a regular grid using a gaussian kernel.
	 -FFT: Fast fourier transform operator.
	 -B: A fourier scaling factor in wave space to transform forces to velocities, see eq.9 in [1].

  sqrt(Mw)·dWw: The far range stochastic contribution is computed in fourier space along M·F as:
       Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
       Only one St·FFTi is needed, the stochastic term is added as a velocity in fourier space.
       dWw is a gaussian random vector of complex numbers, special care must be taken to ensure the correct conjugacy properties needed for the FFT. See pse_ns::fourierBrownianNoise

Therefore, in the case of Mdot_far, for computing M·F, Bw·dWw is also summed.

computeBdW computes only the real space stochastic contribution.

Notes:
Storing F and G functions in r^2 scale (a.i. table(r^2) = F(sqrt(r^2))) creates artifacts due to the linear interpolation of a cuadratic scale, so it is best to just store table(sqrt(r^2)) = F(r). The cost of the sqrt seems neglegible and yields a really better accuracy.

References:

[1]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations
           -  https://arxiv.org/pdf/1611.09322.pdf
[2]  Spectral accuracy in fast Ewald-based methods for particle simulations
           -  http://www.sciencedirect.com/science/article/pii/S0021999111005092

Special thanks to Marc Melendez and Florencio Balboa.
*/
#ifndef BDHI_PSE_CUH
#define BDHI_PSE_CUH
#include "global/defines.h"
#include "ParticleData/ParticleGroup.cuh"
#include "BDHI.cuh"
#include "PSE/NearField.cuh"
#include "PSE/FarField.cuh"

namespace uammd{
  namespace BDHI{

    class PSE{
    public:
      using Parameters = pse_ns::Parameters;
      PSE(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys,
	  Parameters par);
      
      ~PSE(){}
      
      void setup_step(              cudaStream_t st = 0){}
      /*Compute M·F = Mr·F + Mw·F*/
      void computeMF(real3* MF, cudaStream_t st){
	sys->log<System::DEBUG1>("[BDHI::PSE] Computing MF....");
	int numberParticles = pg->getNumberParticles();
	thrust::fill(thrust::cuda::par.on(st), MF, MF+numberParticles, real3());
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto force = pd->getForce(access::location::gpu, access::mode::read);	
	nearField->Mdot(force.begin(), MF, st);
	farField->computeHydrodynamicDisplacements(pos.begin(), force.begin(), MF, numberParticles, st);
      }

      void computeBdW(real3* BdW, cudaStream_t st){
	sys->log<System::DEBUG2>("[BDHI::PSE] Real space brownian noise");
	//Far contribution is included in farField::Mdot
	nearField->computeBdW(BdW, st);
      }

      void computeDivM(real3* divM, cudaStream_t st = 0){};
      void finish_step(             cudaStream_t st = 0){};

      //Computes both the stochastic and deterministic contributions
      void computeHydrodynamicDisplacements(real4* force, real3* MF, cudaStream_t st = 0){
	int numberParticles = pg->getNumberParticles();
	thrust::fill(thrust::cuda::par.on(st), MF, MF+numberParticles, real3());
	//Compute deterministic part in the near field
	nearField->Mdot(force, MF, st);
	real prefactor = sqrt(2*temperature/dt);
	//Compute stochastic part in the near field
	nearField->computeStochasticDisplacements(MF, prefactor, st);
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	//Compute both deterministic and stochastic part in the far field
	farField->computeHydrodynamicDisplacements(pos.begin(), force, MF, numberParticles, st);
      }

      real getHydrodynamicRadius(){
	return hydrodynamicRadius;
      }
      
      real getSelfMobility(){
	return this->M0;	
      }
      
    private:
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
      shared_ptr<System> sys;      
      real hydrodynamicRadius, M0;
      real temperature, dt;
      std::shared_ptr<pse_ns::NearField> nearField;
      std::shared_ptr<pse_ns::FarField> farField;
    };
  }
}

#include "PSE/initialization.cu"
#endif

/*Raul P. Pelaez 2021. Spectral/Chebyshev Doubly Periodic Stokes Integrator.
 */

#ifndef BDHI_DOUBLYPERIODIC_STOKES_CUH
#define BDHI_DOUBLYPERIODIC_STOKES_CUH

#include "Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"
#include "Integrator/Integrator.cuh"
#include "misc/IBM.cuh"
#include "utils/utils.h"
#include "utils/Box.cuh"
#include "global/defines.h"
namespace uammd{
  namespace BDHI{
    namespace detail{
      
      struct euler_functor{
	real dt;

	euler_functor(real _dt) : dt(_dt) {}

	template<class vec> __host__ __device__
	real4 operator()(vec vel, real4 pos) const {
	  vel.w = 0;
	  return pos + vel*dt;
	}
      };   

    }
    
    class DPStokes: public Integrator{
      int currentStep = 0;
    public:
      using Parameters = DPStokesSlab_ns::DPStokes::Parameters;
      
      DPStokes(shared_ptr<ParticleData> pd, shared_ptr<System> sys, Parameters par):
	DPStokes(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, par){}
      
      DPStokes(shared_ptr<ParticleData> pd, shared_ptr<ParticleGroup> pg, shared_ptr<System> sys, Parameters par):
	Integrator(pd, pg, sys, "BDHI::DPStokes"),
	dt(par.dt){
	dpstokes = std::make_shared<DPStokesSlab_ns::DPStokes>(par);
	CudaSafeCall(cudaStreamCreate(&st));
	CudaCheckError();
      }
    
      ~DPStokes(){
	sys->log<System::MESSAGE>("[DPStokes] Destroyed");
	cudaStreamDestroy(st);
      }

      void forwardTime(){
	computeForces();
	auto velocities = computeParticleDisplacements();
	updatePositions(velocities);
      }
      
    private:
      template<class T> using cached_vector =  DPStokesSlab_ns::cached_vector<T>;
      cudaStream_t st;
      shared_ptr<DPStokesSlab_ns::DPStokes> dpstokes;
      real dt;
      
      void computeForces(){
	sys->log<System::DEBUG2>("[DPStokes] Compute particle forces");
	{
	  auto forces = pd->getForce(access::location::gpu, access::mode::write);
	  thrust::fill(thrust::cuda::par.on(st), forces.begin(), forces.end(), real4());
	}
	for(auto i: interactors){
	  i->updateSimulationTime(currentStep*dt);
	  i->sumForce(st);
	}
	currentStep++;
	CudaCheckError();
      }
      
      cached_vector<real4> computeParticleDisplacements(){
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto force = pd->getForce(access::location::gpu, access::mode::read);
	int numberParticles = pg->getNumberParticles();
	auto velocities = dpstokes->Mdot(pos.begin(), force.begin(), numberParticles, st);
	return velocities;
      }

      void updatePositions(cached_vector<real4> &particleVelsAndPressure){
	sys->log<System::DEBUG2>("[DPStokes] Updating positions");
	auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	thrust::transform(thrust::cuda::par.on(st),
			  particleVelsAndPressure.begin(), particleVelsAndPressure.end(),
			  pos.begin(),
			  pos.begin(),
			  detail::euler_functor(dt));
	CudaCheckError();
      }



    };

  }
}

#endif


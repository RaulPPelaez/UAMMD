
/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver.
 */

#ifndef DOUBLYPERIODIC_STOKES_CUH
#define DOUBLYPERIODIC_STOKES_CUH
#include"uammd.cuh"
#include"Integrator/Integrator.cuh"
#include "System/System.h"
#include "utils/utils.h"
#include "global/defines.h"
#include"misc/ChevyshevUtils.cuh"
#include"StokesSlab/FastChebyshevTransform.cuh"
#include"StokesSlab/utils.cuh"
#include"StokesSlab/Correction.cuh"
#include <memory>
#include <stdexcept>
#include "misc/LanczosAlgorithm.cuh"
#include"StokesSlab/spreadInterp.cuh"

namespace uammd{
  namespace DPStokesSlab_ns{

    class DPStokes{
    public:
      using Grid = chebyshev::doublyperiodic::Grid;
      using WallMode = WallMode;
      //Parameters, -1 means that it will be autocomputed if not present
      struct Parameters{
	int nx, ny;
	int nz = -1;
	real dt;
	real viscosity;
	real Lx, Ly;
	real H;
	real tolerance = 1e-7;
	real w, w_d;
	real hydrodynamicRadius;
	real beta = -1;
	real beta_d = -1;
	real alpha = -1;
	real alpha_d = -1;
	//Can be either none, bottom or slit
	WallMode mode = WallMode::none;
      };

      DPStokes(Parameters par);

      ~DPStokes(){
	System::log<System::MESSAGE>("[DPStokes] Destroyed");
      }


      //Computes the hydrodynamic displacements (velocities) coming from the forces
      // acting on a group of positions.
      template<class PosIterator, class ForceIterator>
      cached_vector<real3> Mdot(PosIterator pos, ForceIterator forces,
				int numberParticles, cudaStream_t st = 0){
	auto M = Mdot(pos, forces, (real4*) nullptr, numberParticles, st);
	return M.first;
      }

      //Computes the linear and angular hydrodynamic displacements (velocities) coming from
      // the forces and torques acting on a group of positions
      //If the torques pointer is null, the function will only compute and return the translational part
      // of the mobility
      template<class PosIterator, class ForceIterator, class TorqueIterator>
      std::pair<cached_vector<real3>, cached_vector<real3>>
      Mdot(PosIterator pos, ForceIterator forces, TorqueIterator torques,
	   int numberParticles, cudaStream_t st){
	cudaDeviceSynchronize();
	System::log<System::DEBUG2>("[DPStokes] Computing displacements");
	auto gridData = ibm->spreadForces(pos, forces, numberParticles, st);
	auto gridForceCheb = fct->forwardTransform(gridData, st);
	if(torques){//Torques are added in Cheb space
	  ibm->addSpreadTorquesFourier(pos, torques, numberParticles, gridForceCheb,
				       fct, st);
	}
	FluidData<complex> fluid = solveBVPVelocity(gridForceCheb, st);
	if(mode != WallMode::none){
	  correction->correctSolution(fluid, gridForceCheb, st);
	}
	cached_vector<real3> particleAngularVelocities;
	if(torques){
	  //Ang. velocities are interpolated from the curl of the velocity, which is
	  // computed in Cheb space.
	  auto gridAngVelsCheb = ibm->computeGridAngularVelocityCheb(fluid, st);
	  auto gridAngVels = fct->inverseTransform(gridAngVelsCheb, st);
	  particleAngularVelocities = ibm->interpolateAngularVelocity(gridAngVels, pos, numberParticles, st);
	}
	gridData = fct->inverseTransform(fluid.velocity, st);
	auto particleVelocities = ibm->interpolateVelocity(gridData, pos, numberParticles, st);
	CudaCheckError();
	return {particleVelocities, particleAngularVelocities};
      }

    private:
      shared_ptr<FastChebyshevTransform> fct;
      shared_ptr<Correction> correction;
      shared_ptr<SpreadInterp> ibm;
      gpu_container<real> zeroModeVelocityChebyshevIntegrals;
      gpu_container<real> zeroModePressureChebyshevIntegrals;

      void setUpGrid(Parameters par);
      void initializeKernel(Parameters par);
      void printStartingMessages(Parameters par);
      void resizeVectors();
      void initializeBoundaryValueProblemSolver();

      void precomputeIntegrals();
      void resetGridForce();
      void tryToResetGridForce();
      FluidData<complex> solveBVPVelocity(DataXYZ<complex> &gridForcesFourier, cudaStream_t st);
      void resizeTmpStorage(size_t size);
      real Lx, Ly;
      real H;
      Grid grid;
      real viscosity;
      real gw;
      real tolerance;
      WallMode mode;
      shared_ptr<BVP::BatchedBVPHandler> bvpSolver;

    };

    namespace detail{
      struct LanczosAdaptor{
	std::shared_ptr<DPStokes> dpstokes;
	real4* pos;
	int numberParticles;

	LanczosAdaptor(std::shared_ptr<DPStokes> dpstokes, real4* pos, int numberParticles):
	  dpstokes(dpstokes),pos(pos),numberParticles(numberParticles){}

	void operator()(real3 *mv, real3* m){
	  auto res = dpstokes->Mdot(pos, m, numberParticles);
	  thrust::copy(thrust::cuda::par, res.begin(), res.end(), mv);
	}

      };

       struct SaruTransform{
	uint s1, s2;
	SaruTransform(uint s1, uint s2):s1(s1), s2(s2){}

	__device__ real3 operator()(uint id){
	  Saru rng(s1, s2, id);
	  return make_real3(rng.gf(0.0f,1.0f), rng.gf(0.0f,1.0f).x);

	}
      };

      auto fillRandomVectorReal3(int numberParticles, uint s1, uint s2){
	cached_vector<real3> noise(numberParticles);
	auto cit = thrust::make_counting_iterator<uint>(0);
	auto tr = thrust::make_transform_iterator(cit, SaruTransform(s1, s2));
	thrust::copy(tr, tr + numberParticles,
		     noise.begin());
	return noise;
      }

      struct SumPosAndNoise{
	real3* b;
	real4* pos;
	real sign;
	SumPosAndNoise(real4* pos, real3* b, real sign): pos(pos), b(b), sign(sign){}

	__device__ auto operator()(int id){
	  return make_real3(pos[id]) + sign*b[id];
	}
      };

      struct BDIntegrate{
	real4* pos;
	real3* mf;
	real3* sqrtmdw;
	real3* mpw;
	real3* mmw;
	real rfdPrefactor;
	real dt;
	real noisePrefactor;
	BDIntegrate(real4* pos,
		    real3* mf, real3* sqrtmdw,
		    real3* mpw, real3* mmw, real deltaRFD,
		    real dt, real temperature):
	  pos(pos), mf(mf), sqrtmdw(sqrtmdw),
	  mpw(mpw), mmw(mmw), dt(dt){
	  this->noisePrefactor = sqrt(2*temperature*dt);
	  this->rfdPrefactor = dt*temperature/deltaRFD;
	}
	__device__ void operator()(int id){
	  real3 displacement = mf[id]*dt +
	    noisePrefactor*sqrtmdw[id] +
	    rfdPrefactor * (mpw[id] - mmw[id]);
	  pos[id] += make_real4(displacement);
	}
      };

    }

    class DPStokesIntegrator: public Integrator{
      int steps = 0;
      uint seed;
      std::shared_ptr<DPStokes> dpstokes;
      std::shared_ptr<LanczosAlgorithm> lanczos;
      real deltaRFD;
      public:
      template<class T> using cached_vector = cached_vector<T>;
      struct Parameters: DPStokes::Parameters{
	real temperature;
      };

      DPStokesIntegrator(std::shared_ptr<ParticleData> pd, Parameters par):
	Integrator(pd, "DPStokes"), par(par){
	dpstokes = std::make_shared<DPStokes>(par);
	lanczos = std::make_shared<LanczosAlgorithm>(par.tolerance);
	System::log<System::MESSAGE>("[DPStokes] dt %g", par.dt);
	System::log<System::MESSAGE>("[DPStokes] temperature: %g", par.temperature);
	this->seed = pd->getSystem()->rng().next32();
	this->deltaRFD = 1e-4;
      }

      void forwardTime(){
	if(steps==0) setUpInteractors();
	for(auto i: interactors){
	  i->updateSimulationTime(par.dt*steps);
	  i->sum({.force=true});
	}
	auto pos = pd->getPos(access::gpu, access::readwrite);
	auto force = pd->getForceIfAllocated(access::gpu, access::read);
	auto torque = pd->getTorqueIfAllocated(access::gpu, access::read);
	const int numberParticles = pd->getNumParticles();
	auto mf = dpstokes->Mdot(pos.raw(), force.raw(), numberParticles, 0);
	if(torque.raw()){
	  System::log<System::EXCEPTION>("[DPStokes] Torques are not yet implemented");
	  throw std::runtime_error("Operation not implemented");
	}
	cached_vector<real3> bdw(numberParticles);
	thrust::fill(bdw.begin(), bdw.end(), real3());
	detail::LanczosAdaptor dot(dpstokes, pos.raw(), numberParticles);
	auto noise = detail::fillRandomVectorReal3(numberParticles, seed, 2*steps);
	lanczos->solve(dot,
		       (real*)bdw.data().get(), (real*)noise.data().get(),
		       numberParticles, 0);
	auto noise2 = detail::fillRandomVectorReal3(numberParticles, seed, 2*steps+1);
	auto cit = thrust::make_counting_iterator(0);
	auto posp = thrust::make_transform_iterator(cit,
						    detail::SumPosAndNoise(pos.raw(),
									   noise2.data().get(),
									   deltaRFD*0.5));
	auto mpw = dpstokes->Mdot(posp, noise2.data().get(), numberParticles, 0);
	auto posm = thrust::make_transform_iterator(cit,
						    detail::SumPosAndNoise(pos.raw(),
									   noise2.data().get(),
									   -deltaRFD*0.5));
	auto mmw = dpstokes->Mdot(posm, noise2.data().get(), numberParticles, 0);
	detail::BDIntegrate bd(pos.raw(), mf.data().get(), bdw.data().get(),
			    mpw.data().get(), mmw.data().get(),
			    deltaRFD, par.dt, par.temperature);
	thrust::for_each_n(cit, numberParticles, bd);
	steps++;
      }

    private:
      Parameters par;

      void setUpInteractors(){
	Box box({par.Lx, par.Ly,par.H});
	box.setPeriodicity(1, 1, 0);
	for(auto i: interactors){
	  i->updateBox(box);
	  i->updateTimeStep(par.dt);
	  i->updateTemperature(par.temperature);
	  i->updateViscosity(par.viscosity);
	}

      }
    };
  }
}
#include"StokesSlab/initialization.cu"
#include"StokesSlab/DPStokes.cu"
#endif

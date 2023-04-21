
/*Raul P. Pelaez 2019-2021. Spectral/Chebyshev Doubly Periodic Stokes solver.
 */

#ifndef DOUBLYPERIODIC_STOKES_CUH
#define DOUBLYPERIODIC_STOKES_CUH
#include "misc/LanczosAlgorithm/MatrixDot.h"
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
#include<thrust/functional.h>

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

      // compute average velocity in the x direction as a function of z
      template<class PosIterator, class ForceIterator>
      std::vector<double>
      computeAverageVelocity(PosIterator pos, ForceIterator forces,
			     int numberParticles, cudaStream_t st = 0){
      	cudaDeviceSynchronize();
      	System::log<System::DEBUG2>("[DPStokes] Computing displacements");
      	auto gridData = ibm->spreadForces(pos, forces, numberParticles, st);
      	auto gridForceCheb = fct->forwardTransform(gridData, st);
      	FluidData<complex> fluid = solveBVPVelocity(gridForceCheb, st);
      	if(mode != WallMode::none){
      	  correction->correctSolution(fluid, gridForceCheb, st);
      	}
	int nx = this->grid.cellDim.x, ny = this->grid.cellDim.y, nz = this->grid.cellDim.z;
	// std::cout << "printing average velocity" << std::endl;
	auto xvel = fluid.velocity.m_x;
	std::vector<double> averageVelocity(nz);
	std::vector<complex> chebCoeff(nz);
	for(int i=0;i<nz;i++){
	  chebCoeff[i] = xvel[(nx/2+1)*ny*i]/(nx*ny);
	}
	// transfer to real space by direct summation
	// Chebyshev stuff refresher
	// f(z) = c_0+c_1T_1(z)+c_2T_2(z)+...
	// z = (b+a)/2+(b-a)/2*cos(i*M_PI/(nz-1));
	// arg = acos(-1+2*(z-a)/(b-a));
	// T_j(z) = cos(j acos(-1+2*(z-a)/(b-a))) with z = (b+a)/2+(b-a)/2*cos(i*M_PI/(nz-1))
	for(int i=0;i<nz;i++){
	  real arg = i*M_PI/(nz-1);
	  for(int j=0;j<nz;j++){
	    averageVelocity[i] += chebCoeff[j].x*cos(j*arg);
	  }
	}
	
      	CudaCheckError();
      	return averageVelocity;
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
      struct LanczosAdaptor: lanczos::MatrixDot{
	std::shared_ptr<DPStokes> dpstokes;
	real4* pos;
	int numberParticles;

	LanczosAdaptor(std::shared_ptr<DPStokes> dpstokes, real4* pos, int numberParticles):
	  dpstokes(dpstokes),pos(pos),numberParticles(numberParticles){}

	void operator()(real *v, real* mv) override{
	  auto res = dpstokes->Mdot(pos, (real3*)v, numberParticles);
	  thrust::copy(thrust::cuda::par, res.begin(), res.begin() + numberParticles, (real3*)mv);
	}

      };

      struct SaruTransform{
	uint s1, s2;
	real std;
	SaruTransform(uint s1, uint s2, real std = 1.0):
	  s1(s1), s2(s2), std(std){}

	__device__ real3 operator()(uint id){
	  Saru rng(s1, s2, id);
	  return make_real3(rng.gf(0.0f,std), rng.gf(0.0f,std).x);

	}
      };

      auto fillRandomVectorReal3(int numberParticles, uint s1, uint s2, real std = 1.0){
	cached_vector<real3> noise(numberParticles);
	auto cit = thrust::make_counting_iterator<uint>(0);
	auto tr = thrust::make_transform_iterator(cit, SaruTransform(s1, s2, std));
	thrust::copy(thrust::cuda::par,
		     tr, tr + numberParticles,
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
	real3* noise;
	real3* noisePrev;
	real3* rfd;
	real dt;
	BDIntegrate(real4* pos,
		    real3* mf,
		    real3* noise, real3* noisePrev,
		    real3* rfd,
		    real dt, real temperature):
	  pos(pos), mf(mf), noise(noise),noisePrev(noisePrev),
	  rfd(rfd), dt(dt){
	}
	__device__ void operator()(int id){
	  real3 displacement = mf[id]*dt;
	  if(noise){
	    real3 fluct;
	    if(noisePrev)
	      fluct = dt*real(0.5)*(noise[id] + noisePrev[id]);
	    else
	      fluct = dt*noise[id];
	    displacement += fluct + rfd[id];
	  }
	  pos[id] += make_real4(displacement);
	}
      };

    }

    class DPStokesIntegrator: public Integrator{
      int steps = 0;
      uint seed, seedRFD;
      std::shared_ptr<DPStokes> dpstokes;
      std::shared_ptr<lanczos::Solver> lanczos;
      thrust::device_vector<real3> previousNoise;
      real deltaRFD;
      public:
      template<class T> using cached_vector = cached_vector<T>;
      struct Parameters: DPStokes::Parameters{
	real temperature = 0;
	bool useLeimkuhler = false;
      };

      DPStokesIntegrator(std::shared_ptr<ParticleData> pd, Parameters par):
	Integrator(pd, "DPStokes"), par(par){
	dpstokes = std::make_shared<DPStokes>(par);
	lanczos = std::make_shared<lanczos::Solver>();
	System::log<System::MESSAGE>("[DPStokes] dt %g", par.dt);
	System::log<System::MESSAGE>("[DPStokes] temperature: %g", par.temperature);
	this->seed = pd->getSystem()->rng().next32();
	this->seedRFD = pd->getSystem()->rng().next32();
	this->deltaRFD = 1e-6*par.hydrodynamicRadius;
      }

      //Returns the thermal drift term: temperature*dt*(\partial_q \cdot M)
      auto computeThermalDrift(){
	auto pos = pd->getPos(access::gpu, access::read);
	const int numberParticles = pd->getNumParticles();
	if(par.temperature){
	  auto noise2 = detail::fillRandomVectorReal3(numberParticles, seedRFD, steps);
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
	  using namespace thrust::placeholders;
	  thrust::transform(mpw.begin(), mpw.end(),
			    mmw.begin(),
			    mpw.begin(),
			    make_real3(par.dt*par.temperature/deltaRFD)*(_1 - _2));
	  return mpw;
	}
	else{
	  return cached_vector<real3>();
	}
      }

      //Returns sqrt(2*M*temperature/dt)dW
      auto computeFluctuations(){
	auto pos = pd->getPos(access::gpu, access::read);
	const int numberParticles = pd->getNumParticles();
	cached_vector<real3> bdw(numberParticles);
	thrust::fill(bdw.begin(), bdw.end(), real3());
	if(par.temperature){
	  detail::LanczosAdaptor dot(dpstokes, pos.raw(), numberParticles);
	  auto noise = detail::fillRandomVectorReal3(numberParticles, seed, steps,
						     sqrt(2*par.temperature/par.dt));
	  lanczos->run(dot,
		       (real*)bdw.data().get(), (real*)noise.data().get(),
		       par.tolerance, 3*numberParticles);
	}
	return bdw;
      }

      //Returns the product of the forces and the mobility matrix, M F
      auto computeDeterministicDisplacements(){
	auto pos = pd->getPos(access::gpu, access::read);
	auto force = pd->getForce(access::gpu, access::read);
	if(pd->isTorqueAllocated()){
	  System::log<System::EXCEPTION>("[DPStokes] Torques are not yet implemented");
	  throw std::runtime_error("Operation not implemented");
	}
	const int numberParticles = pd->getNumParticles();
    	return dpstokes->Mdot(pos.raw(), force.raw(), numberParticles, 0);
      }

      void forwardTime(){
	System::log<System::DEBUG2>("[DPStokes] Running step %d", steps);
	if(steps==0) setUpInteractors();
	resetForces();
	for(auto i: interactors){
	  i->updateSimulationTime(par.dt*steps);
	  i->sum({.force=true});
	}
	const int numberParticles = pd->getNumParticles();
	auto mf = computeDeterministicDisplacements();
	if(par.temperature){
	  auto bdw = computeFluctuations();
	  if(par.useLeimkuhler and previousNoise.size() != numberParticles){
	    previousNoise.resize(numberParticles);
	    thrust::copy(bdw.begin(), bdw.end(), previousNoise.begin());
	  }
	  auto rfd = computeThermalDrift();
	  real3* d_prevNoise = par.useLeimkuhler?previousNoise.data().get():nullptr;
	  auto pos = pd->getPos(access::gpu, access::readwrite);
	  detail::BDIntegrate bd(pos.raw(), mf.data().get(),
				 bdw.data().get(), d_prevNoise,
				 rfd.data().get(),
				 par.dt, par.temperature);
	  auto cit = thrust::make_counting_iterator(0);
	  thrust::for_each_n(thrust::cuda::par, cit, numberParticles, bd);
	  if(par.useLeimkuhler)
	    thrust::copy(bdw.begin(), bdw.end(), previousNoise.begin());
	}
	else{
	  auto pos = pd->getPos(access::gpu, access::readwrite);
	  detail::BDIntegrate bd(pos.raw(), mf.data().get(),
				 nullptr, nullptr, nullptr,
				 par.dt, 0);
	  auto cit = thrust::make_counting_iterator(0);
	  thrust::for_each_n(thrust::cuda::par, cit, numberParticles, bd);
	}
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
      void resetForces(){
	auto force = pd->getForce(access::gpu, access::write);
	thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
      }
    };
  }
}
#include"StokesSlab/initialization.cu"
#include"StokesSlab/DPStokes.cu"
#endif

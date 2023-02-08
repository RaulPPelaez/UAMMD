/* Raul P. Pelaez 2022. Compressible Inertial Coupling Method.*/
#include"ICM_Compressible.cuh"
#include <string>
#include <thrust/transform.h>
#include "ICM_Compressible/spreadInterp.cuh"
#include "ICM_Compressible/SpatialDiscretization.cuh"
#include "ICM_Compressible/Fluctuations.cuh"
#include "ICM_Compressible/FluidSolver.cuh"
namespace uammd{
  namespace Hydro{

    template<class Walls>
    auto ICM_Compressible_impl<Walls>::storeCurrentPositions(){
      System::log<System::DEBUG2>("[ICM_Compressible] Store current particle positions");
      int numberParticles = pg->getNumberParticles();
      cached_vector<real4> v(numberParticles);
      auto pos = pd->getPos(access::gpu, access::read);
      thrust::copy(pos.begin(), pos.end(), v.begin());
      return v;
    }

    template<class Walls>
    auto ICM_Compressible_impl<Walls>::interpolateFluidVelocityToParticles(const DataXYZ &fluidVelocity){
      System::log<System::DEBUG2>("[ICM_Compressible] Interpolate fluid velocities");
      using namespace icm_compressible;
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::gpu, access::read);
      auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
      auto vel = staggered::interpolateFluidVelocities(fluidVelocity, pos.begin(),
						       kernel, numberParticles, grid);
      return vel;
    }

    template<class Walls>
    auto ICM_Compressible_impl<Walls>::spreadCurrentParticleForcesToFluid(){
      System::log<System::DEBUG2>("[ICM_Compressible] Spread particle forces");
      using namespace icm_compressible;
      auto forces = pd->getForce(access::gpu, access::read);
      auto pos = pd->getPos(access::gpu, access::read);
      auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
      int numberParticles = pg->getNumberParticles();
      auto fluidForcing = staggered::spreadParticleForces(forces.begin(), pos.begin(),
							  kernel, numberParticles, grid);
      return fluidForcing;
    }

    namespace detail{
      real  getRK3SubStepTime(int subStep){
	real time = 0;
	switch(subStep){
	case 1:
	  time = 1.0/3.0;
	  break;
	case 2:
	  time = 2.0/3.0;
	  break;
	case 3:
	  time = 1;
	  break;
	default:
	  throw std::runtime_error("Invalid RK3 sub step");
	};
	return time;
      }
    }

    template<class Walls>
    template<int subStep>
    auto ICM_Compressible_impl<Walls>::callRungeKuttaSubStep(const DataXYZ &fluidForcingAtHalfStep,
							     const cached_vector<real2> &fluidStochasticTensor,
							     FluidPointers fluidAtSubTime){
      System::log<System::DEBUG2>("[ICM_Compressible] Runge Kutta sub step %d", subStep);
      using namespace icm_compressible;
      FluidData fluidAtNewTime(getGhostGridSize());
      FluidTimePack fluid{currentFluid.getPointers(), fluidAtSubTime, fluidAtNewTime.getPointers()};
      FluidParameters params{shearViscosity, bulkViscosity, dt};
      callRungeKuttaSubStepGPU<subStep>(grid,
					fluid,
					DataXYZPtr(fluidForcingAtHalfStep),
					thrust::raw_pointer_cast(fluidStochasticTensor.data()),
					params, *densityToPressure, steps);
      auto density_ptr = thrust::raw_pointer_cast(fluidAtNewTime.density.data());
      real subStepTime = detail::getRK3SubStepTime(subStep); // 1/3, 2/3, 1
      for(auto i: updatables) i->updateSimulationTime((steps+subStepTime)*dt);
      fillGhostCells(fluidAtNewTime.getPointers());
      callMomentumToVelocityGPU(getGridSize(), fluidAtNewTime.getPointers());
      fillGhostCells(fluidAtNewTime.getPointers());
      return fluidAtNewTime;
    }

    template<class Walls>
    void ICM_Compressible_impl<Walls>::fillGhostCells(FluidPointers fluid){
      System::log<System::DEBUG2>("[ICM_Compressible] Updating ghost cells");
      icm_compressible::callUpdateGhostCells(fluid, walls, ghostCells, grid.cellDim);
    }

    //Uses the RK3 solver in FluidSolver.cuh
    template<class Walls>
    void ICM_Compressible_impl<Walls>::updateFluidWithRungeKutta3(const DataXYZ &fluidForcingAtHalfStep,
						      const cached_vector<real2> &fluidStochasticTensor){
      System::log<System::DEBUG2>("[ICM_Compressible] Update fluid with RK3");
      auto fluidPrediction = callRungeKuttaSubStep<1>(fluidForcingAtHalfStep, fluidStochasticTensor,
						      currentFluid.getPointers());
      auto fluidAtHalfStep = callRungeKuttaSubStep<2>(fluidForcingAtHalfStep, fluidStochasticTensor,
						      fluidPrediction.getPointers());
      fluidPrediction.clear();
           fluidPrediction = callRungeKuttaSubStep<3>(fluidForcingAtHalfStep, fluidStochasticTensor,
						      fluidAtHalfStep.getPointers());
      currentFluid.density.swap(fluidPrediction.density);
      currentFluid.velocity.swap(fluidPrediction.velocity);
      currentFluid.momentum.swap(fluidPrediction.momentum);
    }
    template<class Walls>
    auto ICM_Compressible_impl<Walls>::computeStochasticTensor(){
      System::log<System::DEBUG2>("[ICM_Compressible] Compute stochastic tensor");
      using namespace icm_compressible;
      cached_vector<real2> fluidStochasticTensor;
      if(temperature > 0){
	auto cellDim = getGhostGridSize();
	int numberCells = cellDim.x*cellDim.y*cellDim.z;
	fluidStochasticTensor.resize(randomNumbersPerCell*numberCells);
	auto fluidStochasticTensor_ptr = thrust::raw_pointer_cast(fluidStochasticTensor.data());
	FluidParameters params{shearViscosity, bulkViscosity, dt};
	callFillStochasticTensorGPU(grid,
				    fluidStochasticTensor_ptr,
				    seed, uint(steps), params, temperature);
	icm_compressible::callUpdateGhostCellsFluctuations(fluidStochasticTensor, walls, ghostCells, grid.cellDim);
      }
      return fluidStochasticTensor;
    }

    template<class Walls>
    void ICM_Compressible_impl<Walls>::forwardFluidDensityAndVelocityToNextStep(const DataXYZ &fluidForcingAtHalfStep){
      System::log<System::DEBUG2>("[ICM_Compressible] Forward fluid to next step");
      auto fluidStochasticTensor = computeStochasticTensor();
      updateFluidWithRungeKutta3(fluidForcingAtHalfStep, fluidStochasticTensor);
    }

    template<class Walls>
    void ICM_Compressible_impl<Walls>::updateParticleForces(){
      System::log<System::DEBUG2>("[ICM_Compressible] Compute particle forces");
      {
	auto force = pd->getForce(access::gpu, access::write);
	thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
      }
      for(auto i: interactors) i->sum({.force=true});
    }

    //Any external fluid forcing (for instance a shear flow) can be added here.
    //The solver assumes the external forcing remains constant throughout the timestep and requires the forces at time n+1/2
    template<class Walls>
    void ICM_Compressible_impl<Walls>::addFluidExternalForcing(DataXYZ &fluidForcingAtHalfStep){
      // thrust::transform(fluidForcingAtHalfStep.x(),
      // 			fluidForcingAtHalfStep.x() + fluidForcingAtHalfStep.size(),
      // 			fluidForcingAtHalfStep.x(),
      // 			[=]__device__(real fx){ return fx +=1; });
    }

    template<class Walls>
    auto ICM_Compressible_impl<Walls>::computeCurrentFluidForcing(){
      System::log<System::DEBUG2>("[ICM_Compressible] Compute fluid forcing");
      //Fluid forcing is required at half step.
      for(auto i: updatables) i->updateSimulationTime((steps+0.5)*dt);
      updateParticleForces();
      auto fluidForcing = spreadCurrentParticleForcesToFluid();
      addFluidExternalForcing(fluidForcing);
      return fluidForcing;
    }

    namespace icm_compressible{

      //Particle temporal integrator (Euler predictor-corrector)
      struct MidStepEulerFunctor{

	real dt;
	MidStepEulerFunctor(real dt):dt(dt){}

	__device__ auto operator()(real4 p, real3 v){
	  return make_real4(make_real3(p)+real(0.5)*dt*v, p.w);
	}
      };

      auto sumVelocities(const DataXYZ &v1, const DataXYZ &v2){
	int size = v1.size();
	DataXYZ v3(size);
	thrust::transform(thrust::cuda::par, v1.x(), v1.x() + size, v2.x(), v3.x(), thrust::plus<real>());
	thrust::transform(thrust::cuda::par, v1.y(), v1.y() + size, v2.y(), v3.y(), thrust::plus<real>());
	thrust::transform(thrust::cuda::par, v1.z(), v1.z() + size, v2.z(), v3.z(), thrust::plus<real>());
	return v3;
      }
    }

    //Takes positions to n+1/2: \vec{q}^{n+1/2} = \vec{q}^n + dt/2\oper{J}^n\vec{v}^n
    template<class Walls>
    void ICM_Compressible_impl<Walls>::forwardPositionsToHalfStep(){
      if(pg->getNumberParticles() > 0){
	System::log<System::DEBUG2>("[ICM_Compressible] Forward particles to n+1/2");
	auto velocities = interpolateFluidVelocityToParticles(currentFluid.velocity);
	auto pos = pd->getPos(access::gpu, access::readwrite);
	thrust::transform(thrust::cuda::par,
			  pos.begin(), pos.end(), velocities.xyz(),
			  pos.begin(),
			  icm_compressible::MidStepEulerFunctor(dt));
      }
    }

    //Takes positions to n+1: \vec{q}^{n+1} = \vec{q}^n + dt/2\oper{J}^{n+1/2}(\vec{v}^n + \vec{v}^{n+1})
    template<class Walls>
    void ICM_Compressible_impl<Walls>::forwardPositionsToNextStep(const cached_vector<real4> &positionsAtN,
						      const DataXYZ &fluidVelocitiesAtN){
      System::log<System::DEBUG2>("[ICM_Compressible] Forward particles to n+1");
      auto fluidVelocitiesAtMidStep = icm_compressible::sumVelocities(fluidVelocitiesAtN, currentFluid.velocity);
      auto velocities = interpolateFluidVelocityToParticles(fluidVelocitiesAtMidStep);
      auto pos = pd->getPos(access::gpu, access::readwrite);
      thrust::transform(thrust::cuda::par,
			positionsAtN.begin(), positionsAtN.end(),
			velocities.xyz(),
			pos.begin(),
			icm_compressible::MidStepEulerFunctor(dt));
    }

    template<class Walls>
    void ICM_Compressible_impl<Walls>::forwardTime(){
      System::log<System::DEBUG>("[ICM_Compressible] Forward time");
      auto positionsAtN = storeCurrentPositions();
      auto fluidVelocitiesAtN = currentFluid.velocity;
      forwardPositionsToHalfStep();
      {
	auto fluidForcing = computeCurrentFluidForcing();
	forwardFluidDensityAndVelocityToNextStep(fluidForcing);
      }
      forwardPositionsToNextStep(positionsAtN, fluidVelocitiesAtN);
      steps++;
    }

  }
}

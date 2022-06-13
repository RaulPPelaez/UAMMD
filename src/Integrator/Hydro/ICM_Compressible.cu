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

    auto ICM_Compressible::storeCurrentPositions(){
      System::log<System::DEBUG2>("[ICM_Compressible] Store current particle positions");
      int numberParticles = pg->getNumberParticles();
      ICM_Compressible::cached_vector<real4> v(numberParticles);
      auto pos = pd->getPos(access::gpu, access::read);
      thrust::copy(pos.begin(), pos.end(), v.begin());
      return v;
    }

    auto ICM_Compressible::interpolateFluidVelocityToParticles(const DataXYZ &fluidVelocity){
      System::log<System::DEBUG2>("[ICM_Compressible] Interpolate fluid velocities");
      using namespace icm_compressible;
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::gpu, access::read);
      auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
      auto vel = staggered::interpolateFluidVelocities(fluidVelocity, pos.begin(), kernel, numberParticles, grid);
      return vel;
    }

    auto ICM_Compressible::spreadCurrentParticleForcesToFluid(){
      System::log<System::DEBUG2>("[ICM_Compressible] Spread particle forces");
      using namespace icm_compressible;
      auto forces = pd->getForce(access::gpu, access::read);
      auto pos = pd->getPos(access::gpu, access::read);
      auto kernel = std::make_shared<Kernel>(grid.cellSize.x);
      int numberParticles = pg->getNumberParticles();
      auto fluidForcing = staggered::spreadParticleForces(forces.begin(), pos.begin(), kernel, numberParticles, grid);
      return fluidForcing;
    }

    template<class Container>
    void checkDensity(Container &dens, int3 n){
      std::vector<real> h_d(dens.size(), 0);
      thrust::copy(dens.begin(), dens.end(), h_d.begin());
      int jj= 0;
      fori(0, h_d.size()){
	int3 cell = icm_compressible::getCellFromThreadId(i, n+2);
	//int ii = icm_compressible::linearIndexGhost3D(cell, n+2);
	if(h_d[i]<=0.0){
	  std::cerr<<"Density is bad at "<<i<<": "<<cell.x<<" "<<cell.y<<" "<<cell.z<<": "<<h_d[i]<<std::endl;
	  jj++;
	  if(jj==5) break;
	}
      }
    }

    template<int subStep>
    auto ICM_Compressible::callRungeKuttaSubStep(const DataXYZ &fluidForcingAtHalfStep,
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
      icm_compressible::callUpdateGhostCells(thrust::raw_pointer_cast(fluidAtNewTime.density.data()),
					     ghostCells, grid.cellDim);
      callMomentumToVelocityGPU(getGridSize(), fluidAtNewTime.getPointers());
      fillGhostCells(fluidAtNewTime.getPointers());
      //      checkDensity(fluidAtNewTime.density, grid.cellDim);
      return fluidAtNewTime;
    }

    void ICM_Compressible::fillGhostCells(FluidPointers fluid){
      System::log<System::DEBUG2>("[ICM_Compressible] Updating ghost cells");
      icm_compressible::callUpdateGhostCells(fluid, ghostCells, grid.cellDim);
    }

    //Uses the RK3 solver in FluidSolver.cuh
    void ICM_Compressible::updateFluidWithRungeKutta3(const DataXYZ &fluidForcingAtHalfStep,
						      const cached_vector<real2> &fluidStochasticTensor){
      System::log<System::DEBUG2>("[ICM_Compressible] Update fluid with RK3");
      fillGhostCells(currentFluid.getPointers());
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

    auto ICM_Compressible::computeStochasticTensor(){
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
	icm_compressible::callUpdateGhostCellsFluctuations(fluidStochasticTensor, ghostCells, grid.cellDim);
      }
      return fluidStochasticTensor;
    }

    void ICM_Compressible::forwardFluidDensityAndVelocityToNextStep(const DataXYZ &fluidForcingAtHalfStep){
      System::log<System::DEBUG2>("[ICM_Compressible] Forward fluid to next step");
      auto fluidStochasticTensor = computeStochasticTensor();
      updateFluidWithRungeKutta3(fluidForcingAtHalfStep, fluidStochasticTensor);
    }

    void ICM_Compressible::updateParticleForces(){
      System::log<System::DEBUG2>("[ICM_Compressible] Compute particle forces");
      {
	auto force = pd->getForce(access::gpu, access::write);
	thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
      }
      for(auto i: interactors) i->sum({.force=true});
    }

    //Any external fluid forcing (for instance a shear flow) can be added here.
    //The solver assumes the external forcing remains constant throughout the timestep and requires the forces at time n+1/2
    void ICM_Compressible::addFluidExternalForcing(DataXYZ &fluidForcingAtHalfStep){
      // thrust::transform(fluidForcingAtHalfStep.x(),
      // 			fluidForcingAtHalfStep.x() + fluidForcingAtHalfStep.size(),
      // 			fluidForcingAtHalfStep.x(),
      // 			[=]__device__(real fx){ return fx +=1; });
    }

    auto ICM_Compressible::computeCurrentFluidForcing(){
      System::log<System::DEBUG2>("[ICM_Compressible] Compute fluid forcing");
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
    void ICM_Compressible::forwardPositionsToHalfStep(){
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
    void ICM_Compressible::forwardPositionsToNextStep(const cached_vector<real4> &positionsAtN,
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

    void ICM_Compressible::forwardTime(){
      System::log<System::DEBUG>("[ICM_Compressible] Forward time");
      auto positionsAtN = storeCurrentPositions();
      auto fluidVelocitiesAtN = currentFluid.velocity;
      forwardPositionsToHalfStep();
      for(auto i: interactors) i->updateSimulationTime((steps+0.5)*dt);
      {
	auto fluidForcing = computeCurrentFluidForcing();
	forwardFluidDensityAndVelocityToNextStep(fluidForcing);
      }
      forwardPositionsToNextStep(positionsAtN, fluidVelocitiesAtN);
      steps++;
      for(auto i: interactors) i->updateSimulationTime(steps*dt);
    }

  }
}

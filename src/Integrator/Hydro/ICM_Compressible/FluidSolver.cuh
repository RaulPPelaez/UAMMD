/*Raul P. Pelaez 2022. Compressible Inertial Coupling Method. Methods for the
fluid temporal integration

The Compressible Navier-Stokes equations can be written as:
    \partial_t \rho = -\nabla\cdot\vec{g}
    \partial_t\vec{g} = -\nabla\cdot(\vec{g}\otimes\vec{v}) -
\nabla\cdot\tens{\sigma} + \vec{f} + \nabla\cdot\tens{Z}

  Where:
    \rho(\vec{r},t): Fluid density
    \vec{v}(\vec{r},t): Fluid velocity
    \vec{g}=\rho\vec{g}: Fluid momentum
    \tens{\sigma} = \nabla\pi - \eta\nabla^2\vec{v} -
(\xi+\eta/3)\nabla(\nabla\cdot\vec{v}): Stress tensor \pi(\rho): Pressure, given
by a provided equation of state (for instance \pi=c_t^2\rho) \vec{f}: Any
external fluid forcing, typically \oper{S}\vec{F}, the spreaded particle forces
    \vec{g}\otimes\vec{v}: I call this the kinectic tensor
    \tens{Z}: Fluctuating stress tensor

Both of the Navier-Stokes equations can be written as a conservation equation
with the following form: U^c = A*U^a + B(U^b + \Delta U(U^b, W^c))

Where U might be the density or the fluid velocity and (a,b,c) are three
different time points inside a time step In order to go from the time step n to
n+1 the solver must be called three times for the density and then the velocity:
 1- a=0, b=n and c=n+1/3
 2- a=3/4, b=1/4 and c=n+2/2
 3- a=1/3, b=2/3 and c=n+1

In both cases, we can define \Delta U = -dt\nabla\cdot\tens{F} + dt\vec{f}
Where \tens{F} means one thing or another depending on the equation we are
solving, \vec{f} is only non-zero for the velocity.

W^c represents the fluctuations, which are defined as:

W^{n+1/3} = W_A- \sqrt(3)W_B
W^{n+2/3} = W_A+ \sqrt(3)W_B
W^{n+1} = W_A
Where W_A and W_B are uncorrelated.
The solver is described in Appendix A of [1]

Other substepping schemes might be used with slight modifications to this code (see Florencio Balboa's Ph.D manuscript)
References:

[1] Inertial coupling for point particle fluctuating hydrodynamics. F. Balboa et. al. 2013
 */
#ifndef ICM_COMPRESSIBLE_TEMPORALDISCRETIZATION_CUH
#define ICM_COMPRESSIBLE_TEMPORALDISCRETIZATION_CUH
#include"uammd.cuh"
#include "utils.cuh"
#include "SpatialDiscretization.cuh"
#include"Fluctuations.cuh"
namespace uammd{
  namespace Hydro{
    namespace icm_compressible{

      //Returns -nabla\cdot\vec{g}dt. Being \vec{g} = \rho\vec{v} the fluid momentum.
      __device__ real computeDensityIncrement(int3 cell_i, FluidPointers fluid, real dt, Grid grid){
	const real3 h = grid.getCellSize(cell_i);
	const auto n = grid.cellDim;
	const real momentumDivergence = computeMomentumDivergence(cell_i, n, fluid, h);
	return -momentumDivergence*dt;
      }

      //Computes the divergence of the stress tensor \tens{\sigma} (without fluctuations),
      // -\nabla\cdot\tens{\sigma} = -\nabla\pi + \eta\nabla^2\vec{v} + (\xi + \eta/3)\nabla(\nabla\cdot\vec{v})
      //  \pi: pressure, coming from a provided equation of state
      //  \xi and \eta are the bulk and shear viscosities
      template<class EquationOfState>
      __device__ real3 computeDeterministicStressDivergence(int3 cell_i, FluidPointers fluid,
							    FluidParameters par, EquationOfState densityToPressure,
							    Grid grid){
	const int3 n = grid.cellDim;
	const real3 h = grid.getCellSize(cell_i);
	const real3 pressureGradient = computePressureGradient(cell_i, n, h, densityToPressure, fluid.density);
	const real3 velocityLaplacian = computeVelocityLaplacian(cell_i, n, h, fluid);
	const real3 velocityDivergenceGradient = computeVelocityDivergenceGradient(cell_i, n, h, fluid);
	const real3 diffusion = par.shearViscosity*velocityLaplacian;
	const real3 convection = (par.bulkViscosity + par.shearViscosity/real(3.0))*velocityDivergenceGradient;
	const real3 internalStress = -pressureGradient;
	return internalStress + diffusion + convection;
      }

      //Returns -(\nabla\cdot(\vec{g}\otimes\vec{v}) + \nabla\cdot\tens{\sigma})dt
      //Here \vec{g} = \rho\vec{v} is the fluid momentum.
      //\tens{\sigma} is the non-fluctuating stress tensor
      //Fluctuations are not included here.
      template<class EquationOfState>
      __device__ real3 computeDeterministicMomentumIncrement(int3 cell_i, FluidPointers fluid,
							     FluidParameters par, EquationOfState densityToPressure,
							     Grid grid){
	real3 momentumIncrement = real3();
	momentumIncrement += computeKineticDerivative(cell_i, fluid, grid);
	momentumIncrement -= computeDeterministicStressDivergence(cell_i, fluid, par, densityToPressure, grid);
	return -par.dt*momentumIncrement;
      }

      class RungeKutta3{
	real timePrefactorA, timePrefactorB;
	auto setTimePrefactors(int subStep){
	  if(subStep == 1){
	    timePrefactorA = 0;
	    timePrefactorB = 1;
	  }
	  else if(subStep == 2){
	    timePrefactorA = real(3.0)/real(4.0);
	    timePrefactorB = real(0.25);
	  }
	  else if(subStep == 3){
	    timePrefactorA = real(1.0)/real(3.0);
	    timePrefactorB = real(2.0)/real(3.0);
	  }
	}

      public:
	RungeKutta3(int subStep){
	  setTimePrefactors(subStep);
	}

	__device__ real incrementScalar(real scalarAtTimeA, real scalarAtTimeB, real increment) const{
	  return timePrefactorA*scalarAtTimeA + timePrefactorB*(scalarAtTimeB + increment);
	}
      };

     //Solves U^c = A*U^a + B(U^b + \Delta U(U^b))
      //Where U might be the density or the fluid velocity and (a,b,c) are three different time points inside a time step
      //In order to go from the time step n to n+1 the solver must be called three times for the density and then the velocity:
      // 1- a=0, b=n and c=n+1/3
      // 2- a=3/4, b=1/4 and c=n+2/2
      // 3- a=1/3, b=2/3 and c=n+1
      template<int subStep, class EquationOfState>
      __global__ void rungeKuttaSubStepD(FluidTimePack fluid, DataXYZPtr fluidForcingAtHalfStep,
					 real2* fluidStochasticTensor,
					 FluidParameters par, EquationOfState densityToPressure,
					 Grid grid,
					 RungeKutta3 rk){
	const int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id>=grid.getNumberCells()) return;
	const auto cell_i = getCellFromThreadId(id, grid.cellDim);
	real densityAtTimeC;
	{
	  const real densityIncrement = computeDensityIncrement(cell_i, fluid.timeB, par.dt, grid);
	  densityAtTimeC = rk.incrementScalar(fluid.timeA.density[id], fluid.timeB.density[id], densityIncrement);
	}
	real3 momentumIncrement;
	{
	  const real3 externalForcing = fluidForcingAtHalfStep.xyz()[id];
	  const real3 deterministicMomentumIncrement = computeDeterministicMomentumIncrement(cell_i,
											     fluid.timeB, par,
											     densityToPressure, grid);
	  //Only compute fluctuations if temperature >0, which is encoded as the pointer for the flucutations being non-null
	  real3 fluctuatingMomentumIncrement = real3();
	  if(fluidStochasticTensor){
	    fluctuatingMomentumIncrement = computeFluctuatingStressDivergence<subStep>(cell_i, fluidStochasticTensor, grid);
	  }
	  momentumIncrement = deterministicMomentumIncrement + fluctuatingMomentumIncrement + par.dt*externalForcing;
	}
	real3 momentumAtTimeC;
	const int3 n = grid.cellDim;
	const real3 momentumAtTimeA = computeMomentum(cell_i, n, fluid.timeA);
	const real3 momentumAtTimeB = computeMomentum(cell_i, n, fluid.timeB);
	momentumAtTimeC.x = rk.incrementScalar(momentumAtTimeA.x, momentumAtTimeB.x,  momentumIncrement.x);
	momentumAtTimeC.y = rk.incrementScalar(momentumAtTimeA.y, momentumAtTimeB.y,  momentumIncrement.y);
	momentumAtTimeC.z = rk.incrementScalar(momentumAtTimeA.z, momentumAtTimeB.z,  momentumIncrement.z);
	//Time C and time B or A might be aliased, so wait until the end to modify.
	//Density is needed to compute momentum, so wait until here before updating.
	fluid.timeC.density[id] = densityAtTimeC;
	//Wait until the last moment to update velocity in memory.
	//Store momentum in these arrays, just after this kernel they will be transformed to velocities
	//We cannot do it now because all the densities at time C need to be available.
	fluid.timeC.velocityX[id] = momentumAtTimeC.x;
	fluid.timeC.velocityY[id] = momentumAtTimeC.y;
	fluid.timeC.velocityZ[id] = momentumAtTimeC.z;
      }

      template<int subStep, class ...T>
      void callRungeKuttaSubStepGPU(Grid grid, T...args){
	int threads = 128;
	int blocks = grid.getNumberCells()/threads+1;
	RungeKutta3 rk(subStep);
	rungeKuttaSubStepD<subStep><<<blocks, threads>>>(args..., grid, rk);
      }

    }
  }
}
#endif

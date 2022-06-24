/* Raul P. Pelaez 2022. Compressible Inertial Coupling Method.

In the compressible inertial coupling method we employ a staggered grid for the
spatial discretisation of the Navier-Stokes equations.

Particles dynamics are integrated via a predictor-corrector Euler scheme (forces
are only computed once). By default, the particle-fluid coupling is mediated via
a three point Peskin kernel.

The algorithm is described in detail in Appendix A of [1] or in [3]. Check the
files under ICM_Compressible for detailed information about the solver.

This solver is triply periodic, although walls and such could be included.

USAGE:

//Assume an instance of ParticleData exists
//auto pd = std::make_shared<ParticleData>(numberParticles);
//...

using namespace ICM = Hydro::ICM_Compressible;
ICM::Parameters par;
par.shearViscosity = 1.0;
par.bulkViscosity = 1.0;
par.speedOfSound = 16; //For the equation of state
par.temperature = 0;
//par.hydrodynamicRadius = 1.0; //Particle hydrodynamic radius (used to determine the number of fluid cells)
par.cellDim = {32,32,32}; //Number of fluid cells, if set the hydrodynamicRadius is ignored
par.dt = 0.1;
par.boxSize = {32,32,32}; //Simulation domain
par.seed = 1234; //0 will take a value from the UAMMD generator
//The initial fluid density and velocity can be customized:
par.initialDensity = [](real3 position){return 1.0;};
par.initialVelocityX = [](real3 position){return sin(2*M_PI*position.y);};
par.initialVelocityY = [](real3 position){return 1.0;};
par.initialVelocityZ = [](real3 position){return 1.0;};

auto compressible = std::make_shared<ICM>(pd, par);

//Now use it as any other integrator module
//compressible->addInteractor...
//compressible->forwardTime();
//...


FAQ:

1- I want to fiddle with the boundary conditions:
    -Check the function pbc_cells and fetchScalar in file ICM_Compressible/utils.cuh, which handles what happens when trying to access the information of a cell
    -You can also influence the solver itself (for instance to define special rules for the surfaces of the domain) in the functions of the file FluidSolver.cuh.

2- I want to chenge the spreading kernel:
    -Change the line "using Kernel" below to the type of your kernel. You might also have to change the initialization in the spreading and interpolation functions in ICM_Compressible.cu. You will also have to change the relation between the hydrodynamic radius and the number of fluid cells, do this in the ICM_Compressible constructor.

3- I want to add some special fluid forcing:
    -The function addFluidExternalForcing in ICM_Compressible.cu was created for this.

4- I want to change the equation of state:
    -Check the struct DensityToPressure below.

References:

   [1] Inertial coupling for point particle fluctuating hydrodynamics. F. Balboa et. al. 2013
   [2] STAGGERED SCHEMES FOR FLUCTUATING HYDRODYNAMICS. F. Balboa et. al. 2012
   [3] Ph.D. manuscript. Florencio Balboa.
 */
#ifndef UAMMD_ICM_COMPRESSIBLE_CUH
#define UAMMD_ICM_COMPRESSIBLE_CUH
#include "uammd.cuh"
#include "misc/ParameterUpdatable.h"
#include "Integrator/Integrator.cuh"
#include "misc/IBM_kernels.cuh"
#include "ICM_Compressible/utils.cuh"
#include "ICM_Compressible/SpatialDiscretization.cuh"
#include"ICM_Compressible/GhostCells.cuh"
#include <functional>
#include <memory>
// #ifndef __CUDACC_EXTENDED_LAMBDA__
// #error "This code requires the CUDA flag --extended-lambda to be enabled"
//#endif
namespace uammd{
  namespace Hydro{

    namespace icm_compressible{
      //Equation of state
      struct DensityToPressure{
	real isothermalSpeedOfSound = 1.0;
	__device__ real operator()(real density){
	  return isothermalSpeedOfSound*isothermalSpeedOfSound*density;
	}
      };

      //This class hanldes the walls (if present) in the Z direction
      //Default walled behavior is no walls, which translates into a periodic in the three directions.
      //Note that the walls can be ParameterUpdatable
      class DefaultWalls: public ParameterUpdatable{
	real currentTime = 0;
	real bottomWallvx = 0;
      public:
	//Returns wether there are walls in the Z direction. If false the Z domain ghost cells are periodic.
	__host__ __device__ static constexpr bool isEnabled(){
	  return false;
	}

	//Applies the boundary conditions at the top z wall for the fluid
	__device__ void applyBoundaryConditionZBottom(FluidPointers fluid, int3 ghostCell, int3 n) const{
	  // const int ighost = ghostCell.x + (ghostCell.y + ghostCell.z*(n.y+2))*(n.x+2);
	  // //The index of the cell above the ghost cell
	  // const int ighostZp1 = ghostCell.x + (ghostCell.y + (ghostCell.z+1)*(n.y+2))*(n.x+2);
	  // real rho = fluid.density[ighostZp1];
	  // fluid.density[ighost] = rho;
	  // fluid.velocityX[ighost] = 2*bottomWallvx-fluid.velocityX[ighostZp1];
	  // fluid.velocityY[ighost] = -fluid.velocityY[ighostZp1];
	  // fluid.velocityZ[ighost] = -fluid.velocityZ[ighostZp1];
	  // fluid.momentumX[ighost] = 2*bottomWallvx*rho-fluid.momentumX[ighostZp1];
	  // fluid.momentumY[ighost] = -fluid.momentumY[ighostZp1];
	  // fluid.momentumZ[ighost] = -fluid.momentumZ[ighostZp1];
	}

	//Applies the boundary conditions at the bottom z wall for the fluid
	__device__ static void applyBoundaryConditionZTop(FluidPointers fluid, int3 ghostCell, int3 n){
	  // const int ighost = ghostCell.x + (ghostCell.y + ghostCell.z*(n.y+2))*(n.x+2);
	  // //The index of the cell below the ghost cell
	  // const int ighostZm1 = ghostCell.x + (ghostCell.y + (ghostCell.z-1)*(n.y+2))*(n.x+2);

	  // fluid.density[ighost] = fluid.density[ighostZm1];
	  // fluid.velocityX[ighost] = -fluid.velocityX[ighostZm1];
	  // fluid.velocityY[ighost] = -fluid.velocityY[ighostZm1];
	  // fluid.velocityZ[ighost] = -fluid.velocityZ[ighostZm1];
	  // fluid.momentumX[ighost] = -fluid.momentumX[ighostZm1];
	  // fluid.momentumY[ighost] = -fluid.momentumY[ighostZm1];
	  // fluid.momentumZ[ighost] = -fluid.momentumZ[ighostZm1];
	}

	void updateSimulationTime(real newTime) override{
	  this->currentTime = newTime;
	  this->bottomWallvx = sin(2*M_PI*currentTime);
	}
      };

      struct RandGen{
	uint s1, s2;
	real temperature, dV;
	RandGen(uint s1, uint s2, real temperature, real dV):
	  s1(s1), s2(s2), temperature(temperature), dV(dV){}

	__device__ real operator()(int id){
	  Saru rng (s1, s2, id);
	  real x = rng.gf(0,1).x;
	  return sqrt(temperature/dV)*x;
	}
      };

      real3 cellIndex2CenterPos(int i, int3 n, real3 L){
	int3 cell = {i%n.x, (i/n.x)%n.x, i/(n.x*n.y)};
	real3 pos = (make_real3(cell)/make_real3(n)+0.5)*L;
	return pos;
      }

      real3 cell2CenterPos(int3 cell, int3 n, real3 L){
	real3 pos = (make_real3(cell)/make_real3(n)+0.5)*L;
	return pos;
      }

    }

    class ICM_Compressible: public Integrator{
      template<class T>
      using cached_vector = icm_compressible::cached_vector<T>;
      using DataXYZ = icm_compressible::DataXYZ;
      using FluidPointers = icm_compressible::FluidPointers;
      using FluidData = icm_compressible::FluidData;
      using DensityToPressure = icm_compressible::DensityToPressure;
      using Kernel = IBM_kernels::Peskin::threePoint;
      using Walls = icm_compressible::DefaultWalls;
    public:
      struct Parameters{
	real shearViscosity = -1;
	real bulkViscosity = -1;
	real speedOfSound = -1; //For the equation of state
	real temperature = 0;
	real dt = -1;
	real3 boxSize;
	int3 cellDim; //If set the hydrodynamicRadius is ignored
	real hydrodynamicRadius = -1;
	uint seed = 0; //0 will take a value from the UAMMD generator
	std::function<real(real3)> initialDensity;
	std::function<real(real3)> initialVelocityX;
	std::function<real(real3)> initialVelocityY;
	std::function<real(real3)> initialVelocityZ;
      };

      ICM_Compressible(std::shared_ptr<ParticleData> pd, Parameters par):
	Integrator(pd, "ICM::Compressible"){
	densityToPressure = std::make_shared<DensityToPressure>();
	densityToPressure->isothermalSpeedOfSound = par.speedOfSound;
	checkInputValidity(par);
	this->dt = par.dt;
	this->shearViscosity = par.shearViscosity;
	this->bulkViscosity = par.bulkViscosity;
	this->temperature = par.temperature;
	this->seed = (par.seed==0)?sys->rng().next32():par.seed;
	int3 ncells = par.cellDim;
	if(par.hydrodynamicRadius>0){
	  //0.91 only works for Peskin three point
	  ncells = make_int3(par.boxSize/(0.91*par.hydrodynamicRadius));
	}
	else{
	  par.hydrodynamicRadius = 0.91*par.boxSize.x/par.cellDim.x;
	}
	this->grid = Grid(Box(par.boxSize), ncells);
	if(not this->walls)
	  this->walls = std::make_shared<Walls>();
        this->addUpdatable(walls);
	initializeFluid(par);
	printInitialMessages(par);
	setUpGhostCells();
      }

      void forwardTime() override;

      //Returns the number of cells in the fluid grid in each direction without including ghost cells
      int3 getGridSize() const{
	return this->grid.cellDim;
      }

      //Returns the fluid density in GPU memory.
      auto getCurrentDensity() const{
	System::log<System::DEBUG1>("[ICM_Compressible] Returning a copy of the current density");
	return icm_compressible::deghostifyDensity(currentFluid.density, grid.cellDim);
      }

      //Returns the fluid velocity, interpolated to the cell centers. In GPU memory
      auto getCurrentVelocity() const{
	System::log<System::DEBUG1>("[ICM_Compressible] Computing collocated current velocity");
	auto collocatedVelocityGPU = icm_compressible::computeCollocatedVelocity(currentFluid.velocity, grid.cellDim);
	return collocatedVelocityGPU;
      }

    private:
      Grid grid;
      FluidData currentFluid;
      cached_vector<int3> ghostCells; //A list with cells that lie in the halo of the domain grid
      std::shared_ptr<Walls> walls;

      //Returns the number of cells in the fluid grid in each direction including ghost cells
      //The simulation domain is located in the range i_x = 1:(getGhostGridSize().x-3) and similar for y and z.
      int3 getGhostGridSize() const{
	return this->grid.cellDim + make_int3(2,2,2);
      }

      //Throws an exception if some of the provided parameters are invalid
      void checkInputValidity(Parameters par){
	if(par.shearViscosity <= 0) throw std::runtime_error("[ICM_Compressible] Invalid shear viscosity");
	if(par.bulkViscosity <= 0) throw std::runtime_error("[ICM_Compressible] Invalid bulk viscosity");
	if(par.temperature < 0) throw std::runtime_error("[ICM_Compressible] Invalid temperature");
	if(par.dt < 0) throw std::runtime_error("[ICM_Compressible] Invalid dt");
	if(par.speedOfSound <= 0) throw std::runtime_error("[ICM_Compressible] Invalid speed of sound");
	if(par.boxSize.x <= 0) throw std::runtime_error("[ICM_Compressible] Invalid box size");
	if((par.cellDim.x <= 0 and par.hydrodynamicRadius <=0)
	   or (par.cellDim.x > 0 and par.hydrodynamicRadius >0)){
	  throw std::runtime_error("[ICM_Compressible] I need either an hydrodynamic radius or a number of cells");
	}
      }

      void printInitialMessages(Parameters par) const{
	System::log<System::MESSAGE>("[ICM_Compressible] dt: %g", dt);
	System::log<System::MESSAGE>("[ICM_Compressible] shear viscosity: %g", shearViscosity);
	System::log<System::MESSAGE>("[ICM_Compressible] bulk viscosity: %g", bulkViscosity);
	System::log<System::MESSAGE>("[ICM_Compressible] isothermal speed of sound: %g", par.speedOfSound);
	System::log<System::MESSAGE>("[ICM_Compressible] temperature: %g", temperature);
	System::log<System::MESSAGE>("[ICM_Compressible] Box size: %g %g %g", par.boxSize.x, par.boxSize.y, par.boxSize.z);
	System::log<System::MESSAGE>("[ICM_Compressible] Fluid cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
	const real effective_a = 0.91*(grid.cellDim.x/par.boxSize.x);
	const real d0 = par.temperature/(6*M_PI*par.shearViscosity*effective_a);
	System::log<System::MESSAGE>("[ICM_Compressible] Expected long time particle self diffusion coefficient : %g", d0);
	System::log<System::MESSAGE>("[ICM_Compressible] seed: %u", seed);
	const real h =grid.cellSize.x;
	real a_c = par.speedOfSound*par.dt/h;
	real maxvx = *thrust::max_element(currentFluid.velocity.x(), currentFluid.velocity.x() + currentFluid.velocity.size());
	real maxvy = *thrust::max_element(currentFluid.velocity.y(), currentFluid.velocity.y() + currentFluid.velocity.size());
	real maxvz = *thrust::max_element(currentFluid.velocity.z(), currentFluid.velocity.z() + currentFluid.velocity.size());
	real maxv = std::max({maxvx, maxvy, maxvz});
	real a_v = maxv*par.dt/h;
	System::log<System::MESSAGE>("[ICM_Compressible] Advective CFL Numbers: α_c ~ %g; α_v ~ %g", a_c, a_v);
	real maxDensity = *thrust::max_element(currentFluid.density.begin(), currentFluid.density.end());
	real kinematicViscosity = par.shearViscosity/maxDensity;
	real b = kinematicViscosity*par.dt/(h*h);
	real selfDiffusion = par.temperature/(6*M_PI*par.shearViscosity*par.hydrodynamicRadius);
	real b_c = selfDiffusion*par.dt/(h*h);
	System::log<System::MESSAGE>("[ICM_Compressible] Viscous CFL Numbers: β ~ %g; β_c ~ %g", b, b_c);
	System::log<System::MESSAGE>("[ICM_Compressible] Fluid cell Re=α_c/β=%g", a_c/b);
	real Sc = kinematicViscosity/selfDiffusion;
	System::log<System::MESSAGE>("[ICM_Compressible] Schmidt number: %g", Sc);
      }

      //Initializes the fluid variables (velocity, density and momentum)
      void initializeFluid(Parameters par){
	System::log<System::DEBUG>("[ICM_Compressible] Intializing fluid");
	const real defaultDensity = 1.0;
	const real defaultVelocity = 0.0;
	const int3 cellDim = getGhostGridSize();
	const int ncells = cellDim.x*cellDim.y*cellDim.z;
	currentFluid.resize(cellDim);
	std::vector<real> d_h(ncells, defaultDensity);
	std::vector<real> d_vx(ncells, defaultVelocity);
	std::vector<real> d_vy(ncells, defaultVelocity);
	std::vector<real> d_vz(ncells, defaultVelocity);
	for(int i = 0; i<ncells; i++){
	  int3 cell = icm_compressible::getCellFromThreadId(i, cellDim);
	  if(cell.x==0 or cell.x==cellDim.x-1) continue;
	  if(cell.y==0 or cell.y==cellDim.y-1) continue;
	  if(cell.z==0 or cell.z==cellDim.z-1) continue;
	  real3 pos = icm_compressible::cell2CenterPos(cell-make_int3(1,1,1), grid.cellDim, grid.box.boxSize);
	  if(par.initialDensity) d_h[i] = par.initialDensity(pos);
	  if(par.initialVelocityX) d_vx[i] = par.initialVelocityX(pos);
	  if(par.initialVelocityY) d_vy[i] = par.initialVelocityY(pos);
	  if(par.initialVelocityZ) d_vz[i] = par.initialVelocityZ(pos);
	}
	thrust::copy(d_h.begin(), d_h.end(), currentFluid.density.begin());
	thrust::copy(d_vx.begin(), d_vx.end(), currentFluid.velocity.x());
	thrust::copy(d_vy.begin(), d_vy.end(), currentFluid.velocity.y());
	thrust::copy(d_vz.begin(), d_vz.end(), currentFluid.velocity.z());
	fillGhostCells(currentFluid.getPointers());
	callVelocityToMomentumGPU(getGridSize(), currentFluid.getPointers());
	fillGhostCells(currentFluid.getPointers());
      }

      //Fills a vector with a list of ghost cells (a halo just outside of the simulation domain)
      void setUpGhostCells(){
	System::log<System::DEBUG>("[ICM_Compressible] Intializing ghost cells");
	this->ghostCells = icm_compressible::listGhostCells(grid.cellDim);
	fillGhostCells(currentFluid.getPointers());
      }

      auto storeCurrentPositions();
      void forwardPositionsToHalfStep();
      auto computeCurrentFluidForcing();
       void updateParticleForces();
       auto spreadCurrentParticleForcesToFluid();

      auto interpolateFluidVelocityToParticles(const DataXYZ &fluidVelocities);
      void forwardFluidDensityAndVelocityToNextStep(const DataXYZ &fluidForcingAtHalfStep);
      auto computeStochasticTensor();
      void fillGhostCells(FluidPointers fluid);
      void updateFluidWithRungeKutta3(const DataXYZ &fluidForcingAtHalfStep,
				      const cached_vector<real2> &fluidStochasticTensor);
      template<int subStep>
      auto callRungeKuttaSubStep(const DataXYZ &fluidForcingAtHalfStep,
				 const cached_vector<real2> &fluidStochasticTensor,
				 FluidPointers fluidAtSubTime = FluidPointers());
      void addFluidExternalForcing(DataXYZ &fluidForcingAtHalfStep);
      void forwardPositionsToNextStep(const cached_vector<real4> &currentPositions, const DataXYZ &fluidVelocitiesAtN);

      int steps = 0;
      real dt;
      real shearViscosity, bulkViscosity;
      std::shared_ptr<DensityToPressure> densityToPressure;
      real temperature;
      uint seed = 1234;
    };

  }
}
#include"ICM_Compressible.cu"
#endif

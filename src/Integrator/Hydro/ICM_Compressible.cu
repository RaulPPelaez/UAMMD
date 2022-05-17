/* Raul P. Pelaez 2022. Compressible Inertial Coupling Method.
   
In the compressible inertial coupling method we employ a staggered grid for the
spatial discretisation of the Navier-Stokes equations.

In a staggered grid each quantity kind (scalars, vector or tensor elements) is
defined on a different subgrid.
Scalar fields are defined in the cell centers, vector fields in cell faces and
tensor fields are defined at the centers and edges.

Let us denote a certain scalar field with "p", a vector field with "\vec{v}"
(with components v^\alpha ) and a tensor field with "\tens{E}" (with components
E^{\alpha\beta} ).

Say \vec{i}=(i_x, i_y, i_z) represents a cell in the grid, which is centered at
the position \vec{r}_i. Then, the different fields, corresponding to cell
\vec{i} would be defined at the following locations:

  - p_\vec{i} \rightarrow \vec{r}_\vec{i}
  - \vec{v}^\alpha_{\vec{i}} \rightarrow \vec{r}_\vec{i} + h/2\vec{\alpha}
  - \tens{E}^{\alpha\beta}_{\vec{i}} \rightarrow \vec{r}_\vec{i} +
              h/2\vec{\alpha} + h/2\vec{\beta}

Where \vec{\alpha} and \vec{\beta} are the unit vectors in those directions and
h is the size of a cell.

This rules result in the values assigned to a cell sometimes being defined in
strange places. The sketch below represents all the values owning to a certain
cell, \vec{i} (with center defined at ○). Unintuitively, some quantities asigned
to cell \vec{i} lie in the neighbouring cells (represented below is also cell
\vec{i} + (1,0,0)).

            <------h---->
+-----⬒-----▽-----------+  | ○: p (Cell center, at \vec{r}_\vec{i})
|      	    |	       	|    ◨: v^x
|      	    |	       	|  | ⬒: v^y
|     ○	    ◨  	  △    	|
| 	    |  	       	|  | △: E^{xx}
|      	    |		|    ▽: E^{xy}
+-----------+-----------+  |

Naturally, this discretisation requires special handling of the discretized
versions of the (differential) operators.

For instance, multiplying a scalar and a vector requires interpolating the
scalar at the position of the vector (Since the result, being a vector, must be
defined at the vector subgrids).

\vec{g} := p*\vec{v} \rightarrow g^\alpha_\vec{i} = 0.5*(p_{\vec{i}+vec{\alpha}}
+ p_\vec{i})*v^\alpha_\vec{i}

For more information, check out Raul's manuscript.

*/

#include"ICM_Compressible.cuh"
#include"misc/IBM.cuh"
#include "misc/IBM_kernels.cuh"
#include <thrust/transform.h>
#include<thrust/pair.h>
namespace uammd{
  namespace Hydro{

    auto ICM_Compressible::storeCurrentPositions(){
      int numberParticles = pg->getNumberParticles();
      ICM_Compressible::cached_vector<real4> v(numberParticles);
      auto pos = pd->getPos(access::gpu, access::read);
      thrust::copy(pos.begin(), pos.end(), v.begin());
      return v;
    }

    namespace icm_compressible{
      

      struct ShiftTransform{
	real3 shift;
	ShiftTransform(real3 shift):shift(shift){}

	__device__ auto operator()(real4 p){
	  return make_real3(p)-shift;
	}

      };
      
      template<class PositionIterator>
      auto make_shift_iterator(const PositionIterator &positions, real3 shift){
	return thrust::make_transform_iterator(positions, ShiftTransform(shift));
      }

      template<class ParticleIterator, class PositionIterator, class Kernel>
      auto spreadStaggered(const ParticleIterator &particleData, const PositionIterator &positions,
			   std::shared_ptr<Kernel> kernel,
			   int numberParticles, Grid grid){
	DataXYZ particleDataXYZ(particleData, numberParticles);
	DataXYZ gridData(grid.getNumberCells());
	gridData.fillWithZero();
	const real3 h = grid.cellSize;
	IBM<Kernel> ibm(kernel, grid);
	auto posX = make_shift_iterator(positions, {real(0.5)*h.x, 0, 0});
	ibm.spread(posX, particleDataXYZ.x(), gridData.x(), numberParticles);
	auto posY = make_shift_iterator(positions, {0, real(0.5)*h.y, 0});
	ibm.spread(posY, particleDataXYZ.y(), gridData.y(), numberParticles);
	auto posZ = make_shift_iterator(positions, {0, 0, real(0.5)*h.z});
	ibm.spread(posZ, particleDataXYZ.z(), gridData.z(), numberParticles);
	return gridData;
      }

      template<class ParticleIterator, class PositionIterator, class Kernel>
      auto spreadRegular(ParticleIterator &particleData, PositionIterator &positions,
			 std::shared_ptr<Kernel> kernel,
			 int numberParticles, Grid grid){
	DataXYZ particleDataXYZ(particleData, numberParticles);
	DataXYZ gridData(grid.getNumberCells());
	gridData.fillWithZero();
	const real3 h = grid.cellSize;
	IBM<Kernel> ibm(kernel, grid);
	ibm.spread(positions, particleDataXYZ.x(), gridData.x(), numberParticles);
	ibm.spread(positions, particleDataXYZ.y(), gridData.y(), numberParticles);
	ibm.spread(positions, particleDataXYZ.z(), gridData.z(), numberParticles);
	return gridData;
      }

      template<class PositionIterator, class Kernel>
      auto interpolateStaggered(const DataXYZ &gridData, const PositionIterator &positions,
			   std::shared_ptr<Kernel> kernel,
			   int numberParticles, Grid grid){
	DataXYZ particleDataXYZ(numberParticles);
	particleDataXYZ.fillWithZero();
	const real3 h = grid.cellSize;
	IBM<Kernel> ibm(kernel, grid);
	auto posX = make_shift_iterator(positions, {real(0.5)*h.x, 0, 0});
	ibm.gather(posX, particleDataXYZ.x(), gridData.x(), numberParticles);
	auto posY = make_shift_iterator(positions, {0, real(0.5)*h.y, 0});
	ibm.gather(posY, particleDataXYZ.y(), gridData.y(), numberParticles);
	auto posZ = make_shift_iterator(positions, {0, 0, real(0.5)*h.z});
	ibm.gather(posZ, particleDataXYZ.z(), gridData.z(), numberParticles);
	return particleDataXYZ;
      }

      template<class ParticleIterator, class PositionIterator, class Kernel>
      auto interpolateRegular(ParticleIterator &particleData, PositionIterator &positions,
			 std::shared_ptr<Kernel> kernel,
			 int numberParticles, Grid grid){
	DataXYZ particleDataXYZ(particleData, numberParticles);
	DataXYZ gridData(grid.getNumberCells());
	gridData.fillWithZero();
	const real3 h = grid.cellSize;
	IBM<Kernel> ibm(kernel, grid);
	ibm.gather(positions, particleDataXYZ.x(), gridData.x(), numberParticles);
	ibm.gather(positions, particleDataXYZ.y(), gridData.y(), numberParticles);
	ibm.gather(positions, particleDataXYZ.z(), gridData.z(), numberParticles);
	return gridData;
      }

    }

    auto ICM_Compressible::interpolateFluidVelocityToParticles(const DataXYZ &fluidVelocity){
      using namespace icm_compressible;
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::gpu, access::read);
      auto kernel = std::make_shared<IBM_kernels::Peskin::threePoint>(grid.cellSize.x);
      auto vel = interpolateStaggered(fluidVelocity, pos.begin(), kernel, numberParticles, grid);
      return vel;
    }

    namespace icm_compressible{
      enum class subgrid{x,y,z};

      struct FluidTimePack{
	FluidPointers timeA, timeB, timeC;
      };

      struct FluidParameters{
	real shearViscosity, bulkViscosity;
	real dt;
      };

      class RungeKutta{
	real timePrefactorA, timePrefactorB;
	auto setTimePrefactors(int subStep){
	real timePrefactorA, timePrefactorB;
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
	return thrust::make_pair(timePrefactorA, timePrefactorB);
      }

      public:
	RungeKutta(int subStep){
	  setTimePrefactors(subStep);
	}

	__device__ real incrementScalar(real scalarAtTimeA, real scalarAtTimeB, real increment) const{
	  return timePrefactorA*scalarAtTimeA + timePrefactorB*(scalarAtTimeB + increment);
	}
      };

      __device__ int3 getCellFromThreadId(int id, Grid grid){
	const auto n = grid.cellDim;
	const int3 cell = make_int3(id%n.x, (id/n.x)%n.y, id/(n.x*n.y));
	return cell;
      }

      __device__ int linearIndex3D(int3 cell, int3 n){
	return cell.x + (cell.y + cell.z*n.y)*n.x;
      }

      template<subgrid direction>
      __device__ auto getVelocityPointer(FluidPointers fluid){
	if(direction == subgrid::x) return fluid.velocityX;
	if(direction == subgrid::y) return fluid.velocityY;
	if(direction == subgrid::z) return fluid.velocityZ;
      }

      template<subgrid direction>
      __device__ constexpr int3 getSubgridOffset(){
	if(direction == subgrid::x) return {1,0,0};
	if(direction == subgrid::y) return {0,1,0};
	if(direction == subgrid::z) return {0,0,1};
      }

      template<class ScalarIterator>
      __device__ real fetchScalar(ScalarIterator scalar, int3 cell, int3 n){
	int ic = linearIndex3D(cell, n);
	return scalar[ic];
      }

      template<subgrid direction, class ScalarIterator>
      __device__ real interpolateScalar(ScalarIterator scalar, int3 cell, int3 n){
	const auto si = fetchScalar(scalar, cell, n);
	const auto sj = fetchScalar(scalar, cell+getSubgridOffset<direction>(), n);
	return real(0.5)*(si+sj);
      }

      template<subgrid direction>
      __device__ real computeMomentumElement(int3 cell_i, int3 n, FluidPointers fluid){
	auto velocity_ptr = getVelocityPointer<direction>(fluid);
	const real v_alpha = fetchScalar(velocity_ptr, cell_i, n);
	const real momentum_alpha = interpolateScalar<direction>(fluid.density, cell_i, n)*v_alpha;
	return momentum_alpha;
      }

      __device__ real3 computeMomentum(int3 cell_i, int3 n, FluidPointers fluid){
	real3 momentum_i;
	momentum_i.x = computeMomentumElement<subgrid::x>(cell_i, n, fluid);
	momentum_i.y = computeMomentumElement<subgrid::y>(cell_i, n, fluid);
	momentum_i.z = computeMomentumElement<subgrid::z>(cell_i, n, fluid);
	return momentum_i;
      }

      template<subgrid direction>
      __device__ real momentumDivergenceElement(int3 cell_i, int3 n, FluidPointers fluid, real h){
	const real momentum_i = computeMomentumElement<direction>(cell_i, n, fluid);
	const auto shift = getSubgridOffset<direction>();
	const real momentum_im1 = computeMomentumElement<direction>(cell_i-shift, n, fluid);
	return real(1.0)/h*(momentum_i - momentum_im1);
      }

      __device__ real computeMomentumDivergence(int3 cell_i,  int3 n, FluidPointers fluid, real3 h){
	real divergence = real(0.0);
	divergence += momentumDivergenceElement<subgrid::x>(cell_i, n, fluid, h.x);
	divergence += momentumDivergenceElement<subgrid::y>(cell_i, n, fluid, h.y);
	divergence += momentumDivergenceElement<subgrid::z>(cell_i, n, fluid, h.z);
	return divergence;
      }

      __device__ real computeDensityIncrement(int3 cell_i, FluidPointers fluid, real dt, Grid grid){
	const real3 h = grid.getCellSize(cell_i);
	const auto n = grid.cellDim;
	const real momentumDivergence = computeMomentumDivergence(cell_i, n, fluid, h);
	return -momentumDivergence*dt;
      }


      //returns Z^\alpha\beta = g^\alpha v^\beta, with \vec{g} = \rho\vec{v}.
      //The result is interpolated at the position of a tensor element in the staggered grid,
      // Z^\alpha\beta is defined at center(cell_i) + \hat{\alpha}/2 + \hat{\beta}/2
      template<subgrid alpha, subgrid beta>
      __device__ real computeKineticTensorElement(int3 cell_i, int3 n, FluidPointers fluid){
	const auto alphaOffset = getSubgridOffset<alpha>();
	const auto betaOffset = getSubgridOffset<beta>();
	const auto vBeta_ptr = getVelocityPointer<beta>(fluid);
	const real vBeta = fetchScalar(vBeta_ptr, cell_i, n);
	const real momentumAlpha = computeMomentumElement<alpha>(cell_i, n, fluid);
	const real momentumAlpha_pBeta = computeMomentumElement<alpha>(cell_i+betaOffset, n, fluid);
	const real gAlphaAtCorner = real(0.5)*(momentumAlpha + momentumAlpha_pBeta);
	const real vBeta_pAlpha = fetchScalar(vBeta_ptr, cell_i + alphaOffset, n);
	const real vBetaAtCorner = real(0.5)*(vBeta + vBeta_pAlpha);
	const real gAlpha_vBeta = gAlphaAtCorner*vBetaAtCorner;
	return gAlpha_vBeta;
      }

      //The divergence of a tensor \tens{Z} is defined componentwise, the result is a vector: \vec{K} = \nabla\cdot\tens{Z}
      // K^\alpha = \nabla\cdot(Z^x\alpha, Z^y\alpha, Z^z\alpha) = \sum_beta (\partial_\beta Z^\beta\alpha)
      //This function computes \nabla_\beta Z^\beta\alpha given \tens{Z} = (\rho\vec{v})\otimes\vec{v}
      //In a staggered grid tensor components are defined on cell centers and corners, so before multiplying
      // one must interpolate each quantity at the appropiated location.
      template<subgrid alpha, subgrid beta>
      __device__ real kineticTensorDivergenceSubElement(int3 cell_i, int3 n, FluidPointers fluid, real h){
	const real Z_alpha_beta_at_0 = computeKineticTensorElement<alpha, beta>(cell_i, n, fluid);
	const auto betaoffset = getSubgridOffset<beta>();
	const real Z_alpha_beta_at_betam1 = computeKineticTensorElement<alpha, beta>(cell_i-betaoffset, n, fluid);
	const real derivative_beta = real(1.0)/h*(Z_alpha_beta_at_0 - Z_alpha_beta_at_betam1);
	return derivative_beta;
      }

      //The divergence of a tensor Z is defined componentwise, the result is a vector: \vec{K} = \nabla\cdot\tens{Z}
      // K^\alpha = \nabla\cdot(Z^x\alpha, Z^y\alpha, Z^z\alpha) = \sum_beta (\partial_\beta Z^\beta\alpha)
      //This function computes K^\alpha given \tens{Z} = (\rho\vec{v})\otimes\vec{v}
      template<subgrid alpha>
      __device__ real computeKineticDerivativeElement(int3 cell_i, FluidPointers fluid, Grid grid){
	const auto n = grid.cellDim;
	const auto h = grid.getCellSize(cell_i);
	real tensorDivergence = real();
	tensorDivergence += kineticTensorDivergenceSubElement<alpha, subgrid::x>(cell_i, n, fluid, h.x);
	tensorDivergence += kineticTensorDivergenceSubElement<alpha, subgrid::y>(cell_i, n, fluid, h.y);
	tensorDivergence += kineticTensorDivergenceSubElement<alpha, subgrid::z>(cell_i, n, fluid, h.z);
	return tensorDivergence;
      }

      //Returns \vec{K}=\nabla\cdot(\vec{g}\otimes\vec{v}), being \vec{g} = \rho\vec{v} the momentum.
      //The divergence of a tensor is applied elementwise, K^\alpha = \nabla\cdot(g^\alpha\vec{v})
      __device__ real3 computeKineticDerivative(int3 cell_i, FluidPointers fluid, Grid grid){
	real3 kineticDerivative;
	kineticDerivative.x = computeKineticDerivativeElement<subgrid::x>(cell_i, fluid, grid);
	kineticDerivative.y = computeKineticDerivativeElement<subgrid::y>(cell_i, fluid, grid);
	kineticDerivative.z = computeKineticDerivativeElement<subgrid::z>(cell_i, fluid, grid);
	return kineticDerivative;
      }


      template<class EquationOfState>
      __device__ real3 computePressureGradient(int3 cell_i, int3 n, real3 h,
					       EquationOfState densityToPressure, real* density){
	real3 pressureGradient;
	const real pressureAt0 = densityToPressure(fetchScalar(density, cell_i, n));
	{
	  real pressureAt_pX = densityToPressure(fetchScalar(density, cell_i + getSubgridOffset<subgrid::x>(), n));
	  pressureGradient.x = real(1.0)/h.x*(pressureAt_pX - pressureAt0);
	}
	{
	  real pressureAt_pY = densityToPressure(fetchScalar(density, cell_i + getSubgridOffset<subgrid::y>(), n));
	  pressureGradient.y = real(1.0)/h.y*(pressureAt_pY - pressureAt0);
	}
	{
	  real pressureAt_pZ = densityToPressure(fetchScalar(density, cell_i + getSubgridOffset<subgrid::z>(), n));
	  pressureGradient.z = real(1.0)/h.z*(pressureAt_pZ - pressureAt0);
	}
	return pressureGradient;
      }

      //The laplacian of v is defined as (\nabla^2 v)^\alpha = \nabla\cdot(\nabla v^\alpha)
      // In a staggered grid (\nabla^2 v)^\alpha = \sum_\beta ( v^\alpha_{i+\beta} - 2v^\alpha_{i} + v^\alpha_{i-beta})
      //This function computes the element \beta of the sum
      template<subgrid beta, class Iterator>
	__device__ real staggeredVectorLaplacianSubElement(int3 cell_i, int3 n,
							   Iterator vAlpha_ptr, real vAlphaAt0){
	auto betaOffset = getSubgridOffset<beta>();
	real vAlphaAt_pBeta = fetchScalar(vAlpha_ptr, cell_i + betaOffset, n);
	real vAlphaAt_mBeta = fetchScalar(vAlpha_ptr, cell_i - betaOffset, n);
	real Lv_beta = (vAlphaAt_pBeta - real(2.0)*vAlphaAt0 + vAlphaAt_mBeta);
	return Lv_beta;
      }

      //The laplacian of v is defined as (\nabla^2 v)^\alpha = \nabla\cdot(\nabla v^\alpha)
      //This function computes the element \alpha of the velocity laplacian
      template<subgrid alpha>
      __device__ real computeVelocityLaplacianElement(int3 cell_i, int3 n, real h, FluidPointers fluid){
	const auto vAlpha_ptr = getVelocityPointer<alpha>(fluid);
	const real vAlphaAt0 = fetchScalar(vAlpha_ptr, cell_i, n);
	const real quadrature = real(1.0)/(h*h);
	const real Lvx = staggeredVectorLaplacianSubElement<subgrid::x>(cell_i, n, vAlpha_ptr, vAlphaAt0);
	const real Lvy = staggeredVectorLaplacianSubElement<subgrid::y>(cell_i, n, vAlpha_ptr, vAlphaAt0);
	const real Lvz = staggeredVectorLaplacianSubElement<subgrid::z>(cell_i, n, vAlpha_ptr, vAlphaAt0);
	const real Lv = quadrature*(Lvx + Lvy +Lvz);
	return Lv;
      }

      //Computes the laplacian of the velocity at cell i.
      //The laplacian of v is defined as (\nabla^2 v)^\alpha = \nabla\cdot(\nabla v^\alpha)
      __device__ real3 computeVelocityLaplacian(int3 cell_i, int3 n, real3 h, FluidPointers fluid){
	real3 velocityLaplacian;
	velocityLaplacian.x = computeVelocityLaplacianElement<subgrid::x>(cell_i, n, h.x, fluid);
	velocityLaplacian.y = computeVelocityLaplacianElement<subgrid::y>(cell_i, n, h.y, fluid);
	velocityLaplacian.z = computeVelocityLaplacianElement<subgrid::z>(cell_i, n, h.z, fluid);
	return velocityLaplacian;
      }

      //The gradient of the divergence is a vector with components given by:
      //(GDv)^\alpha = 1/h^2\sum_\beta v^\beta_{i+\alpha} - v^\beta_{i+\alpha-\beta} -v^\beta_i + v^\beta_{i-\beta}
      //This function computes the element beta of the sum
      template<subgrid alpha, subgrid beta>
      __device__ real staggeredVectorDivergenceGradientSumElement(int3 cell_i, int3 n, FluidPointers fluid){
	auto betaOffset = getSubgridOffset<beta>();
	auto alphaOffset = getSubgridOffset<alpha>();
	const auto vBeta_ptr = getVelocityPointer<beta>(fluid);
	const real vBetaAt0 = fetchScalar(vBeta_ptr, cell_i, n);
	real vBetaAt_pAlpha = fetchScalar(vBeta_ptr, cell_i + alphaOffset, n);
	real vBetaAt_pAlpha_mBeta = fetchScalar(vBeta_ptr, cell_i - betaOffset + alphaOffset, n);
	real vBetaAt_mBeta = fetchScalar(vBeta_ptr, cell_i - betaOffset, n);
	real GDv_beta = vBetaAt_pAlpha - vBetaAt_pAlpha_mBeta - vBetaAt0  + vBetaAt_mBeta;
	return GDv_beta;
      }

      //The gradient of the divergence is a vector with components given by:
      //(GDv)^\alpha = 1/h_alpha\partial_\alpha( (D\vec{v})_{i+\alpha} - (D\vec{v})_i ).g
      //This function computes the element \alpha, which is a sum with three elements.
      template<subgrid alpha>
      __device__ real computeVelocityDivergenceGradientElement(int3 cell_i, int3 n, real3 h, FluidPointers fluid){
	const real GDvx = staggeredVectorDivergenceGradientSumElement<alpha, subgrid::x>(cell_i, n, fluid);
	const real GDvy = staggeredVectorDivergenceGradientSumElement<alpha, subgrid::y>(cell_i, n, fluid);
	const real GDvz = staggeredVectorDivergenceGradientSumElement<alpha, subgrid::z>(cell_i, n, fluid);
	const real GDv = GDvx/h.x + GDvy/h.y +GDvz/h.z;
	return GDv;
      }

      //The gradient of the divergence is a vector given by:
      //GDv := \nabla (\nabla \cdot \vec{v})
      __device__ real3 computeVelocityDivergenceGradient(int3 cell_i, int3 n, real3 h, FluidPointers fluid){
	real3 velocityDivergenceGradient;
	velocityDivergenceGradient.x = computeVelocityDivergenceGradientElement<subgrid::x>(cell_i, n, h, fluid);
	velocityDivergenceGradient.y = computeVelocityDivergenceGradientElement<subgrid::y>(cell_i, n, h, fluid);
	velocityDivergenceGradient.z = computeVelocityDivergenceGradientElement<subgrid::z>(cell_i, n, h, fluid);
	return velocityDivergenceGradient/h;
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

      template<int subStep>
      __device__ constexpr real getFluctuatingTimeAPrefactor(){
	return 1;
      }

      template<int subStep>
      __device__ real getFluctuatingTimeBPrefactor(){
	if(subStep == 1) return -sqrt(3.0);
	if(subStep == 2) return +sqrt(3.0);
	else return 0;
      }

      template<subgrid alpha, subgrid beta>
      constexpr __device__ int getSymmetric3x3Index(){
	if(alpha == subgrid::x and beta == subgrid::x) return 0;
	if(alpha == subgrid::y and beta == subgrid::y) return 1;
	if(alpha == subgrid::z and beta == subgrid::z) return 2;
	if(alpha == subgrid::x and beta == subgrid::y) return 3;
	if(alpha == subgrid::y and beta == subgrid::x) return 3;
	if(alpha == subgrid::x and beta == subgrid::z) return 4;
	if(alpha == subgrid::z and beta == subgrid::x) return 4;
	if(alpha == subgrid::y and beta == subgrid::z) return 5;
	if(alpha == subgrid::z and beta == subgrid::y) return 5;
      }
      constexpr int randomNumbersPerCell = 6;

      __global__ void fillStochasticTensorD(real2* noise,
					    uint s1, uint s2,
					    FluidParameters par,
					    real temperature, Grid grid){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int n = grid.getNumberCells();
	if(id >= n) return;
	const real dV = grid.getCellVolume();
	real prefactorCross = sqrt((real(2.0)*par.shearViscosity*temperature)/(dV*par.dt));
	real prefactorTrace = sqrt((par.bulkViscosity*temperature)/(real(3.0)*dV*par.dt))-
	  - real(1.0)/real(3.0)*sqrt((real(2.0)*par.shearViscosity*temperature)/(dV*par.dt));
	Saru rng(s1, s2, id);
	constexpr real sq2 = 1.4142135623730950488016887;
	real2 wxx = rng.gf(real(0.0), sq2);
	real2 wyy = rng.gf(real(0.0), sq2);
	real2 wzz = rng.gf(real(0.0), sq2);
	real2 wxy = rng.gf(real(0.0), real(1.0));
	real2 wxz = rng.gf(real(0.0), real(1.0));
	real2 wyz = rng.gf(real(0.0), real(1.0));
	real2 trace = wxx+wyy+wzz;
	noise[n*getSymmetric3x3Index<subgrid::x, subgrid::x>()+id] = prefactorCross*wxx + prefactorTrace*trace;
	noise[n*getSymmetric3x3Index<subgrid::y, subgrid::y>()+id] = prefactorCross*wyy + prefactorTrace*trace;
	noise[n*getSymmetric3x3Index<subgrid::z, subgrid::z>()+id] = prefactorCross*wzz + prefactorTrace*trace;
	noise[n*getSymmetric3x3Index<subgrid::x, subgrid::y>()+id] = prefactorCross*wxy;
	noise[n*getSymmetric3x3Index<subgrid::x, subgrid::z>()+id] = prefactorCross*wxz;
	noise[n*getSymmetric3x3Index<subgrid::y, subgrid::z>()+id] = prefactorCross*wyz;
      }

      template<int subStep, subgrid alpha, subgrid beta>
      __device__ real getFlucutatingTensorElement(int3 cell_i, int3 n, const real2* noise){
	const int id = linearIndex3D(cell_i, n);
	constexpr real timePrefactorA = getFluctuatingTimeAPrefactor<subStep>();
	const real timePrefactorB = getFluctuatingTimeBPrefactor<subStep>();
	real2 W_AB = noise[n.x*n.y*n.z*getSymmetric3x3Index<alpha,beta>() + id];
	return timePrefactorA*W_AB.x + timePrefactorB*W_AB.y;
      }


      template<int subStep, subgrid alpha, subgrid beta>
      __device__ real fluctuatingTensorDivergenceSubElement(int3 cell_i, int3 n,
							    const real2* fluidStochasticTensor, real h){
	const real Z_alpha_beta_at_0 = getFlucutatingTensorElement<subStep, alpha, beta>(cell_i, n,
											 fluidStochasticTensor);
	const auto betaoffset = getSubgridOffset<beta>();
	const real Z_alpha_beta_at_betam1 = getFlucutatingTensorElement<subStep, alpha, beta>(cell_i - betaoffset, n,
											      fluidStochasticTensor);
	const real derivative_beta = real(1.0)/h*(Z_alpha_beta_at_0 - Z_alpha_beta_at_betam1);
	return derivative_beta;
      }
      
      template<int subStep, subgrid alpha>
      __device__ real computeFluctuatingDerivativeElement(int3 cell_i, const real2* fluidStochasticTensor, Grid grid){
	const auto n = grid.cellDim;
	const auto h = grid.getCellSize(cell_i);
	real tensorDivergence = real();
	tensorDivergence += fluctuatingTensorDivergenceSubElement<subStep, alpha,subgrid::x>(cell_i, n, fluidStochasticTensor, h.x);
	tensorDivergence += fluctuatingTensorDivergenceSubElement<subStep, alpha,subgrid::y>(cell_i, n, fluidStochasticTensor, h.y);
	tensorDivergence += fluctuatingTensorDivergenceSubElement<subStep, alpha,subgrid::z>(cell_i, n, fluidStochasticTensor, h.z);
	return tensorDivergence;
      }

      template<int subStep>
      __device__ real3 computeFlucuatingStressDivergence(int3 cell_i, const real2* fluidStochasticTensor, Grid grid){
	const auto h = grid.getCellSize(cell_i);
	real3 DZ;
	DZ.x = computeFluctuatingDerivativeElement<subStep, subgrid::x>(cell_i, fluidStochasticTensor, grid);
	DZ.y = computeFluctuatingDerivativeElement<subStep, subgrid::y>(cell_i, fluidStochasticTensor, grid);
	DZ.z = computeFluctuatingDerivativeElement<subStep, subgrid::z>(cell_i, fluidStochasticTensor, grid);
	return DZ;
      }

      //Returns -\nabla\cdot(\vec{g}\otimes\vec{v}) - \nabla\cdot\tens{\sigma}
      //Fluctuations are not included here.
      template<class EquationOfState>
      __device__ real3 computeDeterministicMomentumIncrement(int3 cell_i, FluidPointers fluid,
							     FluidParameters par, EquationOfState densityToPressure,
							     Grid grid){
	real3 momentumIncrement = real3();
	momentumIncrement += computeKineticDerivative(cell_i, fluid, grid);
	momentumIncrement += computeDeterministicStressDivergence(cell_i, fluid, par, densityToPressure, grid);
	return momentumIncrement;
      }


     //Solves U^c = A*U^a + B(U^b + \Delta U(U^b))
      //Where U might be the density or the fluid velocity and (a,b,c) are three different time points inside a time step
      //In order to go from the time step n to n+1 the solver must be called three times for the density and then the velocity:
      // 1- a=0, b=n and c=\tilde{n+1}
      // 2- a=3/4, b=1/4 and c=n+1/2
      // 3- a=1/3, b=2/3 and c=n+1
      template<int subStep, class EquationOfState>
      __global__ void rungeKuttaSubStepD(FluidTimePack fluid, DataXYZPtr fluidForcingAtHalfStep,
					 real2* fluidStochasticTensor,
					 FluidParameters par, EquationOfState densityToPressure,
					 Grid grid,
					 RungeKutta rk){
	const int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id>=grid.getNumberCells()) return;
	const auto cell_i = getCellFromThreadId(id, grid);
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
	  const real3 fluctuatingMomentumIncrement = computeFlucuatingStressDivergence<subStep>(cell_i,
												fluidStochasticTensor,
											        grid);
	  momentumIncrement = deterministicMomentumIncrement + fluctuatingMomentumIncrement + externalForcing;
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
	//We cannot do it now because all the densities need to be available.
	fluid.timeC.velocityX[id] = momentumAtTimeC.x;
	fluid.timeC.velocityY[id] = momentumAtTimeC.y;
	fluid.timeC.velocityZ[id] = momentumAtTimeC.z;
      }

      template<int subStep, class ...T>
      void callRungeKuttaSubStepGPU(Grid grid, T...args){
	int threads = 128;
	int blocks = grid.getNumberCells()/threads+1;
	RungeKutta rk(subStep);
	rungeKuttaSubStepD<subStep><<<blocks, threads>>>(args..., grid, rk);
      }

      __global__ void momentumToVelocityD(FluidPointers fluid, Grid grid){
	const int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id>=grid.getNumberCells()) return;
	const auto cell_i = getCellFromThreadId(id, grid);
	const auto n = grid.cellDim;
	//We stored the momentum in the velocity variable before
	const real3 momentum = {fluid.velocityX[id], fluid.velocityY[id], fluid.velocityZ[id]};
	const real densityX = interpolateScalar<subgrid::x>(fluid.density, cell_i, n);
	fluid.velocityX[id] = momentum.x/densityX;
	const real densityY = interpolateScalar<subgrid::y>(fluid.density, cell_i, n);
	fluid.velocityY[id] = momentum.y/densityY;
	const real densityZ = interpolateScalar<subgrid::z>(fluid.density, cell_i, n);
	fluid.velocityZ[id] = momentum.z/densityZ;
      }

      template<class ...T>
      void callMomentumToVelocityGPU(Grid grid, T...args){
	int threads = 128;
	int blocks = grid.getNumberCells()/threads+1;
	momentumToVelocityD<<<blocks, threads>>>(args..., grid);
      }

      template<class ...T>
      void callFillStochasticTensorGPU(Grid grid, T...args){
	int threads = 128;
	int blocks = grid.getNumberCells()/threads+1;
        fillStochasticTensorD<<<blocks, threads>>>(args..., grid);
      }

    }

    template<int subStep>
    auto ICM_Compressible::callRungeKuttaSubStep(const DataXYZ &fluidForcingAtHalfStep,
						 cached_vector<real2> &fluidStochasticTensor,
						 FluidPointers fluidAtSubTime){
      using namespace icm_compressible;
      FluidData fluidAtNewTime(grid);
      FluidPointers currentFluid(currentFluidDensity, currentFluidVelocity);
      if(subStep==1)
	fluidAtSubTime = currentFluid;
      FluidTimePack fluid{currentFluid, fluidAtSubTime, fluidAtNewTime.getPointers()};
      FluidParameters params{shearViscosity, bulkViscosity, dt};
      callRungeKuttaSubStepGPU<subStep>(grid,
					fluid,
					DataXYZPtr(fluidForcingAtHalfStep),
					thrust::raw_pointer_cast(fluidStochasticTensor.data()),
					params, *densityToPressure);
      callMomentumToVelocityGPU(grid, fluidAtNewTime.getPointers());
      return fluidAtNewTime;
    }

    void ICM_Compressible::updateFluidWithRungeKutta3(const DataXYZ &fluidForcingAtHalfStep,
						      cached_vector<real2> &fluidStochasticTensor){
      auto fluidPrediction = callRungeKuttaSubStep<1>(fluidForcingAtHalfStep, fluidStochasticTensor);
      auto fluidAtHalfStep = callRungeKuttaSubStep<2>(fluidForcingAtHalfStep,
						      fluidStochasticTensor, fluidPrediction.getPointers());
           fluidPrediction = callRungeKuttaSubStep<3>(fluidForcingAtHalfStep,
						      fluidStochasticTensor, fluidAtHalfStep.getPointers());
      currentFluidDensity.swap(fluidPrediction.density);
      currentFluidVelocity.swap(fluidPrediction.velocity);
    }

    auto ICM_Compressible::computeStochasticTensor(){
      using namespace icm_compressible;
      cached_vector<real2> fluidStochasticTensor(randomNumbersPerCell*grid.getNumberCells());
      auto fluidStochasticTensor_ptr = thrust::raw_pointer_cast(fluidStochasticTensor.data());
      FluidParameters params{shearViscosity, bulkViscosity, dt};
      callFillStochasticTensorGPU(grid,
				  fluidStochasticTensor_ptr,
				  seed, uint(steps), params, temperature);
      return fluidStochasticTensor;
    }

    void ICM_Compressible::forwardFluidDensityAndVelocityToNextStep(const DataXYZ &fluidForcingAtHalfStep){
      auto fluidStochasticTensor = computeStochasticTensor();
      updateFluidWithRungeKutta3(fluidForcingAtHalfStep, fluidStochasticTensor);
    }

    auto ICM_Compressible::spreadCurrentParticleForcesToFluid(){
      using namespace icm_compressible;
      auto forces = pd->getForce(access::gpu, access::read);
      auto pos = pd->getPos(access::gpu, access::read);
      auto kernel = std::make_shared<IBM_kernels::Peskin::threePoint>(grid.cellSize.x);
      int numberParticles = pg->getNumberParticles();
      auto fluidForcing = spreadStaggered(forces.begin(), pos.begin(), kernel, numberParticles, grid);
      return fluidForcing;
    }

    void ICM_Compressible::updateParticleForces(){
      auto force = pd->getForce(access::gpu, access::write);
      thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
      for(auto i: interactors) i->sum({.force=true});
    }

    auto ICM_Compressible::computeCurrentFluidForcing(){
      updateParticleForces();
      auto fluidForcing = spreadCurrentParticleForcesToFluid();
      addFluidExternalForcing(fluidForcing);
      return fluidForcing;
    }

    namespace icm_compressible{

      struct MidStepEulerFunctor{

	real dt;
	MidStepEulerFunctor(real dt):dt(dt){}

	__device__ auto operator()(real4 p, real3 v){
	  return make_real4(make_real3(p)+real(0.5)*dt*v, p.w);
	}       
      };

      auto sumVelocities(DataXYZ &v1, DataXYZ &v2){
	int size = v1.size();
	DataXYZ v3(size);
	thrust::transform(v1.x(), v1.x() + size, v1.x(), v3.x(), thrust::plus<real>());
	thrust::transform(v1.y(), v1.y() + size, v1.y(), v3.y(), thrust::plus<real>());
	thrust::transform(v1.z(), v1.z() + size, v1.z(), v3.z(), thrust::plus<real>());			
	return v3;

      }
    }
    
    void ICM_Compressible::forwardPositionsToHalfStep(){
      auto velocities = interpolateFluidVelocityToParticles(currentFluidVelocity);
      auto pos = pd->getPos(access::gpu, access::readwrite);
      thrust::transform(thrust::cuda::par,
			pos.begin(), pos.end(), velocities.xyz(),
			pos.begin(),
			icm_compressible::MidStepEulerFunctor(dt));
    }

    void ICM_Compressible::forwardPositionsToNextStep(cached_vector<real4> positionsAtN,
						      DataXYZ &fluidVelocitiesAtN){
      auto fluidVelocitiesAtMidStep = icm_compressible::sumVelocities(fluidVelocitiesAtN, currentFluidVelocity);
      auto velocities = interpolateFluidVelocityToParticles(fluidVelocitiesAtMidStep);
      auto pos = pd->getPos(access::gpu, access::readwrite);
      thrust::transform(thrust::cuda::par,
			positionsAtN.begin(), positionsAtN.end(),
			velocities.xyz(),
			pos.begin(),
			icm_compressible::MidStepEulerFunctor(dt));
    }

    void ICM_Compressible::forwardTime(){
      auto positionsAtN = storeCurrentPositions();
      auto fluidVelocitiesAtN = currentFluidVelocity;
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

/*Raul P. Pelaez 2022. Compressible Inertial Coupling Method. Spatial discretization methods
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
  - \tens{E}^{\alpha\beta}_{\vec{i}} \rightarrow \vec{r}_\vec{i}
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
#ifndef ICM_COMPRESSIBLE_SPATIALDISCRETIZATION_CUH
#define ICM_COMPRESSIBLE_SPATIALDISCRETIZATION_CUH
#include"uammd.cuh"
#include"utils.cuh"
namespace uammd{
  namespace Hydro{
    namespace icm_compressible{
      //Functions related to the staggered grid discretization
      namespace staggered{

	//Returns a unit vector in the direction given by the subgrid.
	template<subgrid direction>
	__device__ constexpr int3 getSubgridOffset(){
	  if(direction == subgrid::x) return {1,0,0};
	  if(direction == subgrid::y) return {0,1,0};
	  if(direction == subgrid::z) return {0,0,1};
	}

	//Interpolates an scalar (defined at cell centers) in staggered grid to the position of a vector (cell faces) in the provided direction.
	template<subgrid direction, class ScalarIterator>
	__device__ real interpolateScalar(ScalarIterator scalar, int3 cell, int3 n){
	  const auto si = fetchScalar(scalar, cell, n);
	  const auto sj = fetchScalar(scalar, cell+getSubgridOffset<direction>(), n);
	  return real(0.5)*(si+sj);
	}

	//Returns \vec{g}^\alpha = \rho^\alpha\vec{v}^\alpha in a staggered grid (cell faces)
	template<subgrid direction>
	__device__ real computeMomentumElement(int3 cell_i, int3 n, FluidPointers fluid){
	  auto velocity_ptr = getVelocityPointer<direction>(fluid);
	  const real v_alpha = fetchScalar(velocity_ptr, cell_i, n);
	  const real momentum_alpha = interpolateScalar<direction>(fluid.density, cell_i, n)*v_alpha;
	  return momentum_alpha;
	}

	//Returns (\nabla\cdot\vec{g})^\alpha in a staggered grid. Here \vec{g} = \rho\vec{v} is the fluid momentum
	template<subgrid direction>
	__device__ real momentumDivergenceElement(int3 cell_i, int3 n, FluidPointers fluid, real h){
	  const real momentum_i = computeMomentumElement<direction>(cell_i, n, fluid);
	  const auto shift = getSubgridOffset<direction>();
	  const real momentum_im1 = computeMomentumElement<direction>(cell_i-shift, n, fluid);
	  return real(1.0)/h*(momentum_i - momentum_im1);
	}

	//returns Z^\alpha\beta = g^\alpha v^\beta, with \vec{g} = \rho\vec{v}.
	//The result is interpolated at the position of a tensor element in the staggered grid,
	// Z^\alpha\beta is defined at center(cell_i) + \hat{\alpha}/2 + \hat{\beta}/2
	template<subgrid alpha, subgrid beta>
	__device__ real computeKineticTensorElement(int3 cell_i, int3 n, FluidPointers fluid){
	  const auto alphaOffset = getSubgridOffset<alpha>();
	  const auto betaOffset = getSubgridOffset<beta>();
	  const auto vAlpha_ptr = getVelocityPointer<alpha>(fluid);
	  const real vAlpha = fetchScalar(vAlpha_ptr, cell_i, n);
	  const real momentumBeta = computeMomentumElement<beta>(cell_i, n, fluid);
	  const real momentumBeta_pAlpha = computeMomentumElement<beta>(cell_i+alphaOffset, n, fluid);
	  const real gBetaAtCorner = real(0.5)*(momentumBeta + momentumBeta_pAlpha);
	  const real vAlpha_pBeta = fetchScalar(vAlpha_ptr, cell_i + betaOffset, n);
	  const real vAlphaAtCorner = real(0.5)*(vAlpha + vAlpha_pBeta);
	  const real gBeta_vAlpha = gBetaAtCorner*vAlphaAtCorner;
	  return gBeta_vAlpha;
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

	//The laplacian of v is defined as (\nabla^2 v)^\alpha = \nabla\cdot(\nabla v^\alpha)
	// In a staggered grid (\nabla^2 v)^\alpha = \sum_\beta ( v^\alpha_{i+\beta} - 2v^\alpha_{i} + v^\alpha_{i-beta})
	//This function computes the element \beta of the sum
	template<subgrid beta, class Iterator>
	__device__ real vectorLaplacianSubElement(int3 cell_i, int3 n,
						  Iterator vAlpha_ptr, real vAlphaAt0){
	  auto betaOffset = getSubgridOffset<beta>();
	  real vAlphaAt_pBeta = fetchScalar(vAlpha_ptr, cell_i + betaOffset, n);
	  real vAlphaAt_mBeta = fetchScalar(vAlpha_ptr, cell_i - betaOffset, n);
	  real Lv_beta = (vAlphaAt_pBeta - real(2.0)*vAlphaAt0 + vAlphaAt_mBeta);
	  return Lv_beta;
	}

	//The gradient of the divergence is a vector with components given by:
	//(GDv)^\alpha = 1/h^2\sum_\beta v^\beta_{i+\alpha} - v^\beta_{i+\alpha-\beta} -v^\beta_i + v^\beta_{i-\beta}
	//This function computes the element beta of the sum
	template<subgrid alpha, subgrid beta>
	__device__ real vectorDivergenceGradientSumElement(int3 cell_i, int3 n, FluidPointers fluid){
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

      }

      //Returns \nabla\cdot\vec{g} in a staggered grid. Here \vec{g} = \rho\vec{v} is the fluid momentum
      __device__ real computeMomentumDivergence(int3 cell_i,  int3 n, FluidPointers fluid, real3 h){
	real divergence = real(0.0);
	using namespace staggered;
	divergence += momentumDivergenceElement<subgrid::x>(cell_i, n, fluid, h.x);
	divergence += momentumDivergenceElement<subgrid::y>(cell_i, n, fluid, h.y);
	divergence += momentumDivergenceElement<subgrid::z>(cell_i, n, fluid, h.z);
	return divergence;
      }

      //The gradient of the divergence is a vector with components given by:
      //(GDv)^\alpha = 1/h_alpha\partial_\alpha( (D\vec{v})_{i+\alpha} - (D\vec{v})_i ).g
      //This function computes the element \alpha, which is a sum with three elements.
      template<subgrid alpha>
      __device__ real computeVelocityDivergenceGradientElement(int3 cell_i, int3 n, real3 h, FluidPointers fluid){
	using namespace staggered;
	const real GDvx = vectorDivergenceGradientSumElement<alpha, subgrid::x>(cell_i, n, fluid);
	const real GDvy = vectorDivergenceGradientSumElement<alpha, subgrid::y>(cell_i, n, fluid);
	const real GDvz = vectorDivergenceGradientSumElement<alpha, subgrid::z>(cell_i, n, fluid);
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

      //The laplacian of v is defined as (\nabla^2 v)^\alpha = \nabla\cdot(\nabla v^\alpha)
      //This function computes the element \alpha of the velocity laplacian
      template<subgrid alpha>
      __device__ real computeVelocityLaplacianElement(int3 cell_i, int3 n, real3 h, FluidPointers fluid){
	using namespace staggered;
	const auto vAlpha_ptr = getVelocityPointer<alpha>(fluid);
	const real vAlphaAt0 = fetchScalar(vAlpha_ptr, cell_i, n);
	const real Lvx = vectorLaplacianSubElement<subgrid::x>(cell_i, n, vAlpha_ptr, vAlphaAt0)/h.x;
	const real Lvy = vectorLaplacianSubElement<subgrid::y>(cell_i, n, vAlpha_ptr, vAlphaAt0)/h.y;
	const real Lvz = vectorLaplacianSubElement<subgrid::z>(cell_i, n, vAlpha_ptr, vAlphaAt0)/h.z;
	const real Lv = (Lvx + Lvy +Lvz);
	return Lv;
      }

      //Computes the laplacian of the velocity at cell i.
      //The laplacian of v is defined as (\nabla^2 v)^\alpha = \nabla\cdot(\nabla v^\alpha)
      __device__ real3 computeVelocityLaplacian(int3 cell_i, int3 n, real3 h, FluidPointers fluid){
	real3 velocityLaplacian;
	velocityLaplacian.x = computeVelocityLaplacianElement<subgrid::x>(cell_i, n, h, fluid);
	velocityLaplacian.y = computeVelocityLaplacianElement<subgrid::y>(cell_i, n, h, fluid);
	velocityLaplacian.z = computeVelocityLaplacianElement<subgrid::z>(cell_i, n, h, fluid);
	return velocityLaplacian/h;
      }

      //The divergence of a tensor Z is defined componentwise, the result is a vector: \vec{K} = \nabla\cdot\tens{Z}
      // K^\alpha = \nabla\cdot(Z^x\alpha, Z^y\alpha, Z^z\alpha) = \sum_beta (\partial_\beta Z^\beta\alpha)
      //This function computes K^\alpha given \tens{Z} = (\rho\vec{v})\otimes\vec{v}
      template<subgrid alpha>
      __device__ real computeKineticDerivativeElement(int3 cell_i, FluidPointers fluid, Grid grid){
	const auto n = grid.cellDim;
	const auto h = grid.getCellSize(cell_i);
	real tensorDivergence = real();
	using namespace staggered;
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

      //Returns \vec{g} = \rho\vec{v}, the fluid momentum at cell i
      __device__ real3 computeMomentum(int3 cell_i, int3 n, FluidPointers fluid){
	real3 momentum_i;
	using namespace staggered;
	momentum_i.x = computeMomentumElement<subgrid::x>(cell_i, n, fluid);
	momentum_i.y = computeMomentumElement<subgrid::y>(cell_i, n, fluid);
	momentum_i.z = computeMomentumElement<subgrid::z>(cell_i, n, fluid);
	return momentum_i;
      }

      //Given an equation of state (transforming density to pressure). Computes the gradient of the pressure at cell i in a staggered grid (each component of the result is defined in a different cell face).
      template<class EquationOfState>
      __device__ real3 computePressureGradient(int3 cell_i, int3 n, real3 h,
					       EquationOfState densityToPressure, real* density){
	using namespace staggered;
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

      //Transforms the fluid momentum stored in fluid.velocity into velocities.
      __global__ void momentumToVelocityD(FluidPointers fluid, Grid grid){
	const int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id>=grid.getNumberCells()) return;
	const auto cell_i = getCellFromThreadId(id, grid.cellDim);
	const auto n = grid.cellDim;
	//We stored the momentum in the velocity variable before
	const real3 momentum = {fluid.velocityX[id], fluid.velocityY[id], fluid.velocityZ[id]};
	using namespace staggered;
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

      //Returns the velocity in a collocated grid, interpolating the staggered velocities to cell centers
      __global__ void computeCollocatedVelocityD(DataXYZPtr staggeredVelocity, DataXYZPtr collocatedVelocity, int3 n){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= n.x*n.y*n.z) return;
	auto cell_i = getCellFromThreadId(id, n);
	real vx_np12 = fetchScalar(staggeredVelocity.x(), cell_i, n);
	real vx_nm12 = fetchScalar(staggeredVelocity.x(), cell_i-staggered::getSubgridOffset<subgrid::x>(), n);
	real vx = real(0.5)*(vx_np12 + vx_nm12);
	collocatedVelocity.x()[id] = vx;
	real vy_np12 = fetchScalar(staggeredVelocity.y(), cell_i, n);
	real vy_nm12 = fetchScalar(staggeredVelocity.y(), cell_i-staggered::getSubgridOffset<subgrid::y>(), n);
	real vy = real(0.5)*(vy_np12 + vy_nm12);
	collocatedVelocity.y()[id] = vy;
	real vz_np12 = fetchScalar(staggeredVelocity.z(), cell_i, n);
	real vz_nm12 = fetchScalar(staggeredVelocity.z(), cell_i-staggered::getSubgridOffset<subgrid::z>(), n);
	real vz = real(0.5)*(vz_np12 + vz_nm12);
	collocatedVelocity.z()[id] = vz;
      }

      auto computeCollocatedVelocity(const DataXYZ &staggeredVelocity, int3 n){
	DataXYZ collocatedVelocity(staggeredVelocity.size());
	DataXYZPtr staggeredVelocity_ptr(staggeredVelocity);
	int threads = 128;
	int blocks = staggeredVelocity.size()/threads+1;
	computeCollocatedVelocityD<<<blocks, threads>>>(staggeredVelocity,
							DataXYZPtr(collocatedVelocity),
							n);
	return collocatedVelocity;
      }

    }
  }
}
#endif

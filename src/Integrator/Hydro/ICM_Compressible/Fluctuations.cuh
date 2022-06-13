/*Raul P. Pelaez 2022. Compressible Inertial Coupling Method. Methods for fluid fluctuations.

  The functions in this file are used to compute \nabla\codt\tens{Z}, the divergence of the stress tensor.
  \tens{Z} must have a very specific correlation to comply with fluctuation-dissipation balance, eqs. A2,A3 and A8 in [1] describe it.

  In particular, A8 is encoded here to comply with the solver in FluidSolver.cuh
  References:
  [1] Inertial coupling for point particle fluctuating hydrodynamics. F. Balboa et. al. 2013

 */
#ifndef ICM_COMPRESSIBLE_FLUCTUATIONS_CUH
#define ICM_COMPRESSIBLE_FLUCTUATIONS_CUH
#include"uammd.cuh"
#include "utils.cuh"
#include"SpatialDiscretization.cuh"
namespace uammd{
  namespace Hydro{
    namespace icm_compressible{
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

      template<subgrid alpha, subgrid beta>
      __device__ int getFluctTensIndex(int3 cell_i, int3 n){
	const int id = linearIndex3D(cell_i, n);
	const int3 numberCellsWithGhosts = n + make_int3(2,2,2);
	const int ntot = numberCellsWithGhosts.x*numberCellsWithGhosts.y*numberCellsWithGhosts.z;
	return ntot*getSymmetric3x3Index<alpha,beta>() + id;
      }

      template<int subStep, subgrid alpha, subgrid beta>
      __device__ real getFluctuatingTensorElement(int3 cell_i, int3 n, const real2* noise){
	//cell_i = pbc_cell(cell_i, n);
	constexpr real timePrefactorA = getFluctuatingTimeAPrefactor<subStep>();
	const real timePrefactorB = getFluctuatingTimeBPrefactor<subStep>();
	real2 W_AB = noise[getFluctTensIndex<alpha, beta>(cell_i, n)];
	return timePrefactorA*W_AB.x + timePrefactorB*W_AB.y;
      }

      namespace staggered{
	template<int subStep, subgrid alpha, subgrid beta>
	__device__ real fluctuatingTensorDivergenceSubElement(int3 cell_i, int3 n,
							      const real2* fluidStochasticTensor, real h){
	  const real Z_alpha_beta_at_0 = getFluctuatingTensorElement<subStep, alpha, beta>(cell_i, n,
											   fluidStochasticTensor);
	  const auto betaoffset = getSubgridOffset<beta>();
	  const real Z_alpha_beta_at_betam1 = getFluctuatingTensorElement<subStep, alpha, beta>(cell_i - betaoffset, n,
												fluidStochasticTensor);
	  const real derivative_beta = real(1.0)/h*(Z_alpha_beta_at_0 - Z_alpha_beta_at_betam1);
	  return derivative_beta;
	}
      }

      template<int subStep, subgrid alpha>
      __device__ real computeFluctuatingDerivativeElement(int3 cell_i, const real2* fluidStochasticTensor, Grid grid){
	const auto n = grid.cellDim;
	const auto h = grid.getCellSize(cell_i);
	real tensorDivergence = real();
	using namespace staggered;
	tensorDivergence += fluctuatingTensorDivergenceSubElement<subStep, alpha,subgrid::x>(cell_i, n,
											     fluidStochasticTensor, h.x);
	tensorDivergence += fluctuatingTensorDivergenceSubElement<subStep, alpha,subgrid::y>(cell_i, n,
											     fluidStochasticTensor, h.y);
	tensorDivergence += fluctuatingTensorDivergenceSubElement<subStep, alpha,subgrid::z>(cell_i, n,
											     fluidStochasticTensor, h.z);
	return tensorDivergence;
      }

      template<int subStep>
      __device__ real3 computeFluctuatingStressDivergence(int3 cell_i, const real2* fluidStochasticTensor, Grid grid){
	const auto h = grid.getCellSize(cell_i);
	real3 DZ;
	DZ.x = computeFluctuatingDerivativeElement<subStep, subgrid::x>(cell_i, fluidStochasticTensor, grid);
	DZ.y = computeFluctuatingDerivativeElement<subStep, subgrid::y>(cell_i, fluidStochasticTensor, grid);
	DZ.z = computeFluctuatingDerivativeElement<subStep, subgrid::z>(cell_i, fluidStochasticTensor, grid);
	return DZ;
      }


      __global__ void fillStochasticTensorD(real2* noise,
					    uint s1, uint s2,
					    FluidParameters par,
					    real temperature, Grid grid){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int ntot = grid.getNumberCells();
	if(id >= ntot) return;
	const real dV = grid.getCellVolume();
	real prefactorCross = sqrt((par.dt*real(2.0)*par.shearViscosity*temperature)/(real(2.0)*dV));
	real prefactorTrace = sqrt((par.dt*par.bulkViscosity*temperature)/(real(2.0)*real(3.0)*dV))-
	  - real(1.0)/real(3.0)*sqrt((par.dt*real(2.0)*par.shearViscosity*temperature)/(real(2.0)*dV));
	Saru rng(s1, s2, id);
	constexpr real sq2 = 1.4142135623730950488016887;
	real2 wxx = rng.gf(real(0.0), sq2);
	real2 wyy = rng.gf(real(0.0), sq2);
	real2 wzz = rng.gf(real(0.0), sq2);
	real2 wxy = rng.gf(real(0.0), real(1.0));
	real2 wxz = rng.gf(real(0.0), real(1.0));
	real2 wyz = rng.gf(real(0.0), real(1.0));
	real2 trace = wxx+wyy+wzz;
	const int3 n = grid.cellDim;
	const int3 cell_i = getCellFromThreadId(id, n);
	noise[getFluctTensIndex<subgrid::x, subgrid::x>(cell_i, n)] = prefactorCross*wxx + prefactorTrace*trace;
	noise[getFluctTensIndex<subgrid::y, subgrid::y>(cell_i, n)] = prefactorCross*wyy + prefactorTrace*trace;
	noise[getFluctTensIndex<subgrid::z, subgrid::z>(cell_i, n)] = prefactorCross*wzz + prefactorTrace*trace;
	noise[getFluctTensIndex<subgrid::x, subgrid::y>(cell_i, n)] = prefactorCross*wxy;
	noise[getFluctTensIndex<subgrid::x, subgrid::z>(cell_i, n)] = prefactorCross*wxz;
	noise[getFluctTensIndex<subgrid::y, subgrid::z>(cell_i, n)] = prefactorCross*wyz;
      }

      template<class ...T>
      void callFillStochasticTensorGPU(Grid grid, T...args){
	int threads = 128;
	int blocks = grid.getNumberCells()/threads+1;
        fillStochasticTensorD<<<blocks, threads>>>(args..., grid);
      }


    }
  }
}
#endif

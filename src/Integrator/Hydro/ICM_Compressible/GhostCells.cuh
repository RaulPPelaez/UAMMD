#ifndef ICM_COMPRESSIBLE_GHOSTS_CUH
#define ICM_COMPRESSIBLE_GHOSTS_CUH
#include "utils.cuh"
#include "SpatialDiscretization.cuh"
#include"Fluctuations.cuh"
namespace uammd{
  namespace Hydro{
    namespace icm_compressible{

      __host__ __device__ int mod(int a, int b){
	return ((a %= b) < 0) ? a+b : a;
      }

      //Computes the cell coordinates that correspond to a given ghost cell with periodic boundary conditions.
      //The returned coordinates are in full grid (the fluid grid including ghost cells, size n+2)
      __host__ __device__ int3 computeMirrorGhostCell(int3 ghostCell, int3 n){
	int3 mirrorCell;
	mirrorCell.x = mod((ghostCell.x-1),n.x) + 1;
	mirrorCell.y = mod((ghostCell.y-1),n.y) + 1;
	mirrorCell.z = mod((ghostCell.z-1),n.z) + 1;
	return mirrorCell;
      }

      auto listGhostCells(int3 n){
	std::vector<int3> ghostCells;
	const int numberGhostCells = (n.x+2)*(n.y+2)*(n.z+2) - n.x*n.y*n.z;
	ghostCells.reserve(numberGhostCells);
	for(int id = 0; id<(n.x+2)*(n.y+2)*(n.z+2); id++){
	  const auto cell = getCellFromThreadId(id, n+2);
	  if(cell.x == 0 or cell.x == n.x+1 or
	     cell.y == 0 or cell.y == n.y+1 or
	     cell.z == 0 or cell.z == n.z+1){
	    ghostCells.push_back(cell);
	  }
	}
	cached_vector<int3> d_ghostCells(ghostCells.size());
	thrust::copy(ghostCells.begin(), ghostCells.end(), d_ghostCells.begin());
	System::log<System::DEBUG3>("[ICM_Compressible] Number of ghost cells: %d, expected %d",
				    ghostCells.size(), numberGhostCells);

	return d_ghostCells;
      }

      enum class wall{ztop, zbottom};

      //Returns whether a given ghost cell pertains to the given wall
      template<wall awall>
      __device__ bool isGhostCellAtWall(int3 cell, int3 n){
	if(awall == wall::ztop and cell.z == n.z+1) return true;
	if(awall == wall::zbottom and cell.z == 0) return true;
	return false;
      }

      //Applies periodic boundary conditions to the given ghost cell
      __device__ void periodifyGhostCell(FluidPointers fluid, int3 ghostCell, int3 n){
	const int ighost = ghostCell.x + (ghostCell.y + ghostCell.z*(n.y+2))*(n.x+2);
	const int3 periodicCell = computeMirrorGhostCell(ghostCell, n);
	const int iperiodic = periodicCell.x + (periodicCell.y + periodicCell.z*(n.y+2))*(n.x+2);
	fluid.density[ighost] = fluid.density[iperiodic];
	fluid.velocityX[ighost] = fluid.velocityX[iperiodic];
	fluid.velocityY[ighost] = fluid.velocityY[iperiodic];
	fluid.velocityZ[ighost] = fluid.velocityZ[iperiodic];
	fluid.momentumX[ighost] = fluid.momentumX[iperiodic];
	fluid.momentumY[ighost] = fluid.momentumY[iperiodic];
	fluid.momentumZ[ighost] = fluid.momentumZ[iperiodic];
      }

      template<class Walls>
      __global__ void updateGhostCellsD(FluidPointers fluid, Walls walls, int3* ghostCells,
					int3 n, int numberGhostCells){
        uint id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=numberGhostCells) return;
	int3 ghostCell = ghostCells[id];
	periodifyGhostCell(fluid, ghostCell, n);
      }

      template<class Walls>
      __global__ void updateGhostCellsWallsD(FluidPointers fluid, Walls walls, int3* ghostCells,
					     int3 n, int numberGhostCells){
        uint id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=numberGhostCells) return;
	int3 ghostCell = ghostCells[id];
	if(walls.isEnabled()){
	  if(isGhostCellAtWall<wall::zbottom>(ghostCell, n)){
	    walls.applyBoundaryConditionZBottom(fluid, ghostCell, n);
	  }
	  else if(isGhostCellAtWall<wall::ztop>(ghostCell, n)){
	    walls.applyBoundaryConditionZTop(fluid, ghostCell, n);
	  }
	}
      }

      template<class Container, class Walls>
      void callUpdateGhostCells(FluidPointers fluid, std::shared_ptr<Walls> walls, Container &ghostCells, int3 n){
	int threads = 128;
	int numberGhostCells = ghostCells.size();
	int blocks = numberGhostCells/threads+1;
	auto ghostCells_ptr = thrust::raw_pointer_cast(ghostCells.data());
	updateGhostCellsD<<<blocks, threads>>>(fluid, *walls, ghostCells_ptr, n, numberGhostCells);
	updateGhostCellsWallsD<<<blocks, threads>>>(fluid, *walls, ghostCells_ptr, n, numberGhostCells);
      }

      template<subgrid alpha, subgrid beta>
      __device__ int getFluctTensIndexGhost(int3 cell_i, int3 n){
	const int3 numberCellsWithGhosts = n + make_int3(2,2,2);
	const int id = linearIndexGhost3D(cell_i, numberCellsWithGhosts);
	const int ntot = numberCellsWithGhosts.x*numberCellsWithGhosts.y*numberCellsWithGhosts.z;
	const auto tensorElement = getSymmetric3x3Index<alpha,beta>();
	const int index =  ntot*tensorElement + id;
	return index;
      }

      template<subgrid alpha, subgrid beta>
      __device__ void mirrorFluctuations(real2* fluidStochasticTensor, int3 ghostCell, int3 periodicCell, int3 n){
	int isource = getFluctTensIndexGhost<alpha, beta>(ghostCell, n);
	int idest = getFluctTensIndexGhost<alpha, beta>(periodicCell, n);
	fluidStochasticTensor[isource] = fluidStochasticTensor[idest];
      }

      __device__ void periodifyFluctuatingGhostCell(real2* fluidStochasticTensor, int3 ghostCell, int3 n){
	int3 periodicCell = computeMirrorGhostCell(ghostCell, n);
	mirrorFluctuations<subgrid::x, subgrid::x>(fluidStochasticTensor, ghostCell, periodicCell, n);
	mirrorFluctuations<subgrid::y, subgrid::y>(fluidStochasticTensor, ghostCell, periodicCell, n);
	mirrorFluctuations<subgrid::z, subgrid::z>(fluidStochasticTensor, ghostCell, periodicCell, n);
	mirrorFluctuations<subgrid::x, subgrid::y>(fluidStochasticTensor, ghostCell, periodicCell, n);
	mirrorFluctuations<subgrid::x, subgrid::z>(fluidStochasticTensor, ghostCell, periodicCell, n);
	mirrorFluctuations<subgrid::y, subgrid::z>(fluidStochasticTensor, ghostCell, periodicCell, n);
      }

      template<subgrid alpha, subgrid beta>
      __device__ void wallFluctuations(real2* fluidStochasticTensor, int3 ghostCell, int3 n){
	int isource = getFluctTensIndexGhost<alpha, beta>(ghostCell, n);
	//See eq. 2.126 in Floren's thesis. We double the noise for Dirichlet boundary conditions
	fluidStochasticTensor[isource] *= real(2.0);
      }

      __device__ void fillWallFluctuatingGhostCell(real2* fluidStochasticTensor, int3 ghostCell, int3 n){
	wallFluctuations<subgrid::x, subgrid::x>(fluidStochasticTensor, ghostCell, n);
	wallFluctuations<subgrid::y, subgrid::y>(fluidStochasticTensor, ghostCell, n);
	wallFluctuations<subgrid::z, subgrid::z>(fluidStochasticTensor, ghostCell, n);
	wallFluctuations<subgrid::x, subgrid::y>(fluidStochasticTensor, ghostCell, n);
	wallFluctuations<subgrid::x, subgrid::z>(fluidStochasticTensor, ghostCell, n);
	wallFluctuations<subgrid::y, subgrid::z>(fluidStochasticTensor, ghostCell, n);
      }

      template<class Walls>
      __global__ void updateGhostCellsFluctuationsD(real2* fluidStochasticTensor, Walls walls, int3* ghostCells,
						    int3 n, int numberGhostCells){
        uint id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=numberGhostCells) return;
	int3 ghostCell = ghostCells[id];
	if(walls.isEnabled()){
	  if(isGhostCellAtWall<wall::zbottom>(ghostCell, n) or
	     isGhostCellAtWall<wall::ztop>(ghostCell, n)){
	    fillWallFluctuatingGhostCell(fluidStochasticTensor, ghostCell, n);
	  }
	}
	else{
	  periodifyFluctuatingGhostCell(fluidStochasticTensor, ghostCell, n);
	}
      }

      template<class Container2, class Container, class Walls>
      void callUpdateGhostCellsFluctuations(Container2 &fluidStochasticTensor, std::shared_ptr<Walls> walls,
					    Container &ghostCells, int3 n){
	int threads = 128;
	int numberGhostCells = ghostCells.size();
	int blocks = numberGhostCells/threads+1;
	auto ghostCells_ptr = thrust::raw_pointer_cast(ghostCells.data());
	auto noise_ptr = thrust::raw_pointer_cast(fluidStochasticTensor.data());
	updateGhostCellsFluctuationsD<<<blocks, threads>>>(noise_ptr, *walls, ghostCells_ptr, n, numberGhostCells);
      }

      //Copies the density from the input (which includes ghost cells) to an output without ghost cells.
      __global__ void deghostifyDensityD(const real* inputDensity, real* outputDensity, int3 n){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= n.x*n.y*n.z) return;
	auto cell_i = getCellFromThreadId(id, n);
	auto ighost = linearIndex3D(cell_i, n);
	outputDensity[id] = inputDensity[ighost];
      }

      //Returns the density without ghost cells.
      template<class Container>
      auto deghostifyDensity(const Container &density, int3 n){
        Container outputDensity(n.x*n.y*n.z);
	int threads = 128;
	int blocks = outputDensity.size()/threads+1;
	deghostifyDensityD<<<blocks, threads>>>(thrust::raw_pointer_cast(density.data()),
						thrust::raw_pointer_cast(outputDensity.data()),
						n);
	return outputDensity;
      }

    }
  }
}
#endif

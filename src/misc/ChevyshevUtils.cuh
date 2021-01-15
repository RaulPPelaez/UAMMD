/*Raul P. Pelaez 2019-2021. Some utilities for working with Chebyshev grids.

 */
#ifndef CHEVYSHEVUTILS_CUH
#define CHEVYSHEVUTILS_CUH
#include"global/defines.h"
#include<vector>
#include "utils/Box.cuh"
#include "utils/debugTools.h"
namespace uammd{
  namespace chebyshev{

    inline __host__ __device__ real clencurt(int i, int nz){
      real v = 1;
      if(nz%2==0){
	if(i==0 or i==nz) return 1.0/(nz*nz-1.0);
	for(int k = 1; k <=nz/2-1; k++){
	  v -= 2*cos(2*k*M_PI*i/nz)/(4.0*k*k-1);
	}
	v -= cos(M_PI*i)/(nz*nz-1.0);
      }
      else{
	if(i==0 or i==nz) return 1.0/(nz*nz);
	for(int k = 1; k <=(nz-1)/2; k++){
	  v -= 2*cos(2*k*M_PI*i/nz)/(4.0*k*k-1);
	}
      }
      return 2.0*v/nz;
    }

    namespace doublyperiodic{

      struct QuadratureWeights{
	QuadratureWeights(real H, real cellSizex, real cellSizey, int nz):
	  isCopy(false){
	  CudaSafeCall(cudaMalloc((void **)&clencurtWeights, (nz+1)*sizeof(real)));
	  this->hxhy = cellSizex*cellSizey;
	  std::vector<real> weights(nz+1, 0);
	  for(int i = 0; i<nz; i++){
	    weights[i] = 0.5*H*clencurt(i, nz-1);
	  }
	  CudaSafeCall(cudaMemcpy(clencurtWeights, weights.data(), (nz+1)*sizeof(real), cudaMemcpyHostToDevice));
	}

	//This copy constructor prevents cuda from calling the destructor after a kernel call
	QuadratureWeights(const QuadratureWeights& _orig ) { *this = _orig; isCopy = true; }

	~QuadratureWeights(){
	  if(!isCopy and clencurtWeights){
	    cudaFree(clencurtWeights);
	  }
	}

	template<class Grid>
	inline  __device__ real operator()(int3 cell, const Grid &grid) const{
	  return hxhy*clencurtWeights[cell.z];
	}

      private:
	real hxhy;
	real *clencurtWeights;
	bool isCopy;
      };

      //A square grid in xy, chebyshev points in z
      struct Grid{
	int3 gridPos2CellIndex;
	int3 cellDim;
	real2 cellSize;
	real2 invCellSize;
	Box box;

	Grid(): Grid(Box(), make_int3(0,0,0)){}

	Grid(Box box, real3 minCellSize):
	  Grid(box, make_int3(box.boxSize/minCellSize)){}

	Grid(Box box, real minCellSize):
	  Grid(box, make_real3(minCellSize)){}

	Grid(Box _box, int3 in_cellDim):
	  box(_box),
	  cellDim(in_cellDim){
	  cellSize.x = box.boxSize.x/cellDim.x;
	  cellSize.y = box.boxSize.y/cellDim.y;
	  invCellSize = 1.0/make_real2(cellSize);
	  box.setPeriodicity(1,1,0);
	  gridPos2CellIndex = {1, cellDim.x, cellDim.x*cellDim.y};
	}

	template<class VecType>
	inline __host__ __device__ int3 getCell(const VecType &r) const{
	  real3 pos_inBox = box.apply_pbc(make_real3(r));
	  int cz = int((cellDim.z)*(acos(real(-2.0)*pos_inBox.z/box.boxSize.z)/real(M_PI)));
	  int3 cell = make_int3((pos_inBox.x+real(0.5)*box.boxSize.x)*invCellSize.x,
				(pos_inBox.y+real(0.5)*box.boxSize.y)*invCellSize.y,
				cz);
	  if(cell.x==cellDim.x) cell.x = 0;
	  if(cell.y==cellDim.y) cell.y = 0;
	  return cell;
	}

	inline __host__ __device__ int getCellIndex(const int3 &cell) const{
	  return dot(cell, gridPos2CellIndex);
	}

	inline __host__  __device__ int3 pbc_cell(const int3 &cell) const{
	  int3 cellPBC;
	  cellPBC.x = pbc_cell_coord<0>(cell.x);
	  cellPBC.y = pbc_cell_coord<1>(cell.y);
	  cellPBC.z = cell.z;
	  return cellPBC;
	}

	template<int coordinate>
	inline __host__  __device__ int pbc_cell_coord(int cell) const{
	  int ncells = 0;
	  if(coordinate == 0){
	    ncells = cellDim.x;
	  }
	  if(coordinate == 1){
	    ncells = cellDim.y;
	  }
	  if(coordinate == 2){
	    return (cell<0 or cell>=cellDim.z)?-1:cell;
	  }
	  if(cell <= -1) cell += ncells;
	  else if(cell >= ncells) cell -= ncells;
	  return cell;
	}

	inline __host__ __device__ int getNumberCells() const{ return cellDim.x*cellDim.y*cellDim.z;}

	inline __host__ __device__ real getCellVolume(int3 cell) const{
	  auto cs = getCellSize(cell);
	  return cs.x*cs.y*cs.z;
	}

	inline __host__ __device__ real3 getCellSize(int3 cell) const{
	  // real cellHeight_z = cospi(real(cell.z)/cellDim.z);
	  // real cellHeight_zp1 = cospi(real(cell.z+1)/cellDim.z);
	  // real csz = abs(real(0.5)*box.boxSize.z*(cellHeight_z - cellHeight_zp1));
	  real csz = 0;
	  return {cellSize.x, cellSize.y, csz};
	}

	inline __host__ __device__ real3 distanceToCellCenter(real3 pos, int3 cell){
	  const real centerZ = cellHeight(cell.z);
	  const real3 cellCenterPos = make_real3(real(-0.5)*box.boxSize.x + cellSize.x*(cell.x),
						 real(-0.5)*box.boxSize.y + cellSize.y*(cell.y),
						 centerZ);
	  const auto dist = box.apply_pbc(pos - cellCenterPos);
	  return dist;
	}

	inline __host__ __device__ real cellHeight(int cellz){
	  return real(0.5)*box.boxSize.z*cospi((real(cellz))/(cellDim.z-1));
	}

      };

    }
  }
}
#endif

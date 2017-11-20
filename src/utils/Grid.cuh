/*Raul P.Pelaez 2017. Grid utility.

Given a certain box and a number of cells, subdivides the box in that number of cells and allows to:
    -Obtain the cell in which a position is located (taking into account PBC according to Box).
    -Compute the linear 1D index of a certain cell in the grid
    -Apply PBC to a cell (as in cellx=-1 goes to cellx=cellDim.x-1)

 */
#ifndef GRID_CUH
#define GRID_CUH

#include "Box.cuh"
#include "vector.cuh"

namespace uammd{
  
  struct Grid{
    /*A magic vector that transforms cell coordinates to 1D index when dotted*/
    /*Simply: 1, ncellsx, ncellsx*ncellsy*/
    int3 gridPos2CellIndex;
    
    int3 cellDim; //ncells in each size
    real3 cellSize;
    real3 invCellSize; /*The inverse of the cell size in each direction*/
    Box box;
    Grid(): Grid(Box(), make_int3(0,0,0)){}
    Grid(Box box, int3 cellDim):
	box(box),
	cellDim(cellDim){

	cellSize = box.boxSize/make_real3(cellDim);
	invCellSize = 1.0/cellSize;
	if(box.boxSize.z == real(0.0)) invCellSize.z = 0;
	
	gridPos2CellIndex = make_int3( 1,
				       cellDim.x,
				       cellDim.x*cellDim.y);
	
    }
    template<class VecType>
    inline __host__ __device__ int3 getCell(const VecType &r) const{	
	// return  int( (p+0.5L)/cellSize )
	int3 cell = make_int3((box.apply_pbc(make_real3(r)) + real(0.5)*box.boxSize)*invCellSize);
	//Anti-Traquinazo guard, you need to explicitly handle the case where a particle
	// is exactly at the box limit, AKA -L/2. This is due to the precision loss when
	// casting int from floats, which gives non-correct results very near the cell borders.
	// This is completly neglegible in all cases, except with the cell 0, that goes to the cell
	// cellDim, which is catastrophic.
	//Doing the previous operation in double precision (by changing 0.5f to 0.5) also works, but it is a bit of a hack and the performance appears to be the same as this.
	//TODO: Maybe this can be skipped if the code is in double precision mode
	if(cell.x==cellDim.x) cell.x = 0;
	if(cell.y==cellDim.y) cell.y = 0;
	if(cell.z==cellDim.z) cell.z = 0;
	return cell;
    }

    inline __host__ __device__ int getCellIndex(const int3 &cell) const{
	return dot(cell, gridPos2CellIndex);
    }

    inline __host__  __device__ int3 pbc_cell(const int3 &cell) const{
	int3 cellPBC;
	cellPBC.x = pbc_cell_coord<0>(cell.x);
	cellPBC.y = pbc_cell_coord<1>(cell.y);
	cellPBC.z = pbc_cell_coord<2>(cell.z);
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
	  ncells = cellDim.z;
	}

	if(cell <= -1) cell += ncells;
	else if(cell >= ncells) cell -= ncells;
	return cell;
    }

  };

}

#endif
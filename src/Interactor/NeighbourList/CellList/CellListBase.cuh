/*Raul P. Pelaez 2020. Cell list implementation
  CellList subdivides the simulation box in cubic cells and uses a hash sort
based algorithm to compute a list of particles in each cell, similar to the
algorithm described in [1].


USAGE:
This class does not need any UAMMD structures to work. Can be used as a standalone object.

//Create:
CellListBase cl;
thrust::device_vector<real4> some_positions(numberParticles);
//fill positions
...
Box someBox(make_real3(32,32,32));
int3 ncells = make_int3(32,32,32);
Grid someGrid(someBox, ncells);
//Update the cell list
cl.update(thrust::raw_pointer_cast(some_positions), numberParticles, someGrid);
//Get a list of particles in each cell (which can be used to find neighbours)
auto data = cl.getCellList();

//Can be coupled with a NeighbourContainer to traverse neighbours or used directly.
//Get a NeighbourContainer
CellList_ns::NeigbourContainer nc(cl);
//See NeighbourContainer for more info

Implementation notes:

  I noticed that filling CellStart with an "empty cell" flag was taking a great
amount of time when there are a lot of cells (big boxes or small cut offs). So
instead of filling cellStart with some value and storing the first particle of
each non-empty cell, now cellStart stores the index of the first particle of a
cell plus a certain hash that changes at each construction. This hash allows to
encode "empty cell" as cellStart[icell] < hash instead of cellStart[icell] =
someFixedValueGreaterThanNumberParticles. This makes resetting cellStart
unnecesary. One only has to clean cellStart at first construction and when this
hash is such that hash+numberParticles >= std::numeric_limits<uint>::max().



References:

[1] http://developer.download.nvidia.com/assets/cuda/files/particles.pdf
*/
#ifndef CELLLISTBASE_CUH
#define CELLLISTBASE_CUH
#include"utils/ParticleSorter.cuh"
#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include"utils/debugTools.h"
#include<thrust/device_vector.h>
#include<third_party/managed_allocator.h>
#include<limits>

namespace uammd{
  namespace CellList_ns{
    namespace detail{
      struct ToReal4{
	template<typename T>
	__device__ real4 operator()(const T &v) const{
	  return make_real4(v);
	}
      };
    }

    template<class InputIterator>
    __global__ void fillCellList(InputIterator sortPos,
				 uint *cellStart, int *cellEnd,
				 uint currentValidCell,
				 int *errorFlag,
				 int N, Grid grid){
      uint id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id<N){
	uint icell, icell2;
	icell = grid.getCellIndex(grid.getCell(make_real3(sortPos[id])));
	if(id>0){ /*Shared memory target VVV*/
	  icell2 = grid.getCellIndex(grid.getCell(make_real3(sortPos[id-1])));
	}
	else{
	  icell2 = 0;
	}
	const int ncells = grid.getNumberCells();
	if(icell>=ncells or icell2>=ncells){
	  errorFlag[0] = 1;
	  return;
	}
	if(icell != icell2 or id == 0){
	  cellStart[icell] = id+currentValidCell;
	  if(id>0)
	    cellEnd[icell2] = id;
	}
	if(id == N-1) cellEnd[icell] = N;
      }
    }
  }

  class CellListBase{
  protected:
    thrust::device_vector<uint> cellStart;
    thrust::device_vector<int>  cellEnd;
    managed_vector<int>  errorFlags;
    thrust::device_vector<real4> sortPos;
    uint currentValidCell;
    int currentValidCell_counter;
    ParticleSorter ps;
    Grid grid;
    cudaEvent_t event;

  public:

    CellListBase(){
      System::log<System::DEBUG>("[CellList] Created");
      CudaSafeCall(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      currentValidCell_counter = -1;
      errorFlags.resize(1);
      errorFlags[0] = 0;
      CudaCheckError();
    }

    ~CellListBase(){
      System::log<System::DEBUG>("[CellList] Destroyed");
      cudaEventDestroy(event);
    }

    template<class PositionIterator>
    void update(PositionIterator pos, int numberParticles, Grid in_grid, cudaStream_t st = 0){
      System::log<System::DEBUG1>("[CellList] Updating list");
      if(!isGridValid(in_grid)){
	throw std::runtime_error("CellList encountered an invalid grid and/or cutoff");
      }
      this->grid = in_grid;
      System::log<System::DEBUG2>("[CellList] Using %d %d %d cells", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
      resizeCellListToCurrentGrid();
      updateCurrentValidCell(numberParticles);
      updateOrderAndStoreInSortPos(pos, numberParticles, st);
      fillCellList(st);
      CudaCheckError();
    }

    //This accesor function is part of CellList only, not part of the NeighbourList interface
    //They allow to obtain a reference to the cell list structures to use them outside
    struct CellListData{
      //[all particles in cell 0, all particles in cell 1,..., all particles in cell ncells]
      //cellStart[i] stores the index of the first particle in cell i (in internal index)
      //cellEnd[i] stores the last particle in cell i (in internal index)
      //So the number of particles in cell i is cellEnd[i]-cellStart[i]
      const uint * cellStart;
      const int  * cellEnd;
      const real4 *sortPos;   //Particle positions in internal index
      const int* groupIndex; //Transformation between internal indexes and group indexes
      Grid grid;
      uint VALID_CELL; //A value in cellStart less than VALID_CELL means the cell is empty
    };

    CellListData getCellList(){
      CellListData cl;
      cl.cellStart   =  thrust::raw_pointer_cast(cellStart.data());
      cl.cellEnd     =  thrust::raw_pointer_cast(cellEnd.data());
      cl.sortPos     =  thrust::raw_pointer_cast(sortPos.data());
      const int numberParticles = sortPos.size();
      cl.groupIndex =  ps.getSortedIndexArray(numberParticles);
      cl.grid = grid;
      cl.VALID_CELL = currentValidCell;
      return cl;
    }

  private:

    bool isGridValid(Grid in_grid){
      // const bool is2D = in_grid.box.boxSize.z == real(0.0);
      // if(in_grid.cellDim.x < 3 or in_grid.cellDim.y < 3 or (in_grid.cellDim.z < 3 and not is2D)){
      //  	System::log<System::ERROR>("[CellList] I cannot work with less than 3 cells per dimension!");
      // 	return false;
      // }
      return true;
    }

    void tryToResizeCellListToCurrentGrid(){
      const uint ncells = grid.getNumberCells();
      if(cellStart.size()!= ncells){
	cellStart.clear();
	cellStart.resize(ncells);
        thrust::fill(thrust::cuda::par, cellStart.begin(), cellStart.end(), 0);
      }
      if(cellEnd.size()!= ncells){
	cellEnd.clear();
	cellEnd.resize(ncells);
      }
      CudaCheckError();
    }

    void resizeCellListToCurrentGrid(){
      try{
	tryToResizeCellListToCurrentGrid();
      }
      catch(...){
	System::log<System::ERROR>("[CellList] Raised exception at cell list resize");
	throw;
      }
    }

    void updateCurrentValidCell(uint numberParticles){
      if(numberParticles != sortPos.size()){
	currentValidCell_counter = -1;
      }
      const bool isCounterUninitialized = (currentValidCell_counter < 0);
      const ullint nextStepMaximumValue = ullint(numberParticles)*(currentValidCell_counter+2);
      constexpr ullint maximumStorableValue = ullint(std::numeric_limits<uint>::max())-1ull;
      const bool nextStepOverflows  = (nextStepMaximumValue >= maximumStorableValue);
      if(isCounterUninitialized or nextStepOverflows){
	currentValidCell = numberParticles;
	currentValidCell_counter = 1;
	thrust::fill(thrust::cuda::par, cellStart.begin(), cellStart.end(), 0);
	CudaCheckError();
      }
      else{
	currentValidCell_counter++;
	currentValidCell = uint(numberParticles)*currentValidCell_counter;
      }
    }

    template<class PositionIterator>
    void updateOrderAndStoreInSortPos(PositionIterator &pos, int numberParticles, cudaStream_t st){
      ps.updateOrderByCellHash<Sorter::MortonHash>(pos, numberParticles, grid.box, grid.cellDim, st);
      CudaCheckError();
      sortPos.resize(numberParticles);
      auto transform_it = thrust::make_transform_iterator(pos, CellList_ns::detail::ToReal4());
      ps.applyCurrentOrder(transform_it, sortPos.begin(), numberParticles, st);
      CudaCheckError();
    }

    void fillCellList(cudaStream_t st){
      System::log<System::DEBUG3>("[CellList] fill Cell List, currentValidCell: %d", currentValidCell);
      const int numberParticles = sortPos.size();
      const int Nthreads = 512;
      const int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      int *d_errorFlag = thrust::raw_pointer_cast(errorFlags.data());
      CellList_ns::fillCellList<<<Nblocks, Nthreads, 0, st>>>(thrust::raw_pointer_cast(sortPos.data()),
							      thrust::raw_pointer_cast(cellStart.data()),
							      thrust::raw_pointer_cast(cellEnd.data()),
							      currentValidCell,
							      d_errorFlag,
							      numberParticles,
							      grid);
#ifdef UAMMD_DEBUG
      CudaSafeCall(cudaDeviceSynchronize());
      if(d_errorFlag[0] > 0){
	System::log<System::ERROR>("[CellList] NaN positions found during construction");
       	throw std::overflow_error("CellList encountered NaN positions");
      }
#endif
      CudaCheckError();
    }

  };


}

#endif

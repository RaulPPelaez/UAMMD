/*Raul P. Pelaez 2017. Cell List implementation 
  Cell List uses the sorting functionality of NeighbourList_Base to construct a cell list.
  The data structures that contain the list are:
      cellStart (ncells): Contains the index of the first particle in cell icell, in sorted arrays.
      cellEnd (ncells): Same as above, but contaisn the last particle.
      particleIndex(N): Converts the new order to the old. old_index = particleIndex[ordered_index]
   With this information, the particles in each cell can be found and transversed quickly and close in memory.
   To get all the neighbours of a given particle i, get the cell of particle i, and go through the particles in all the 27 neighbour cells.
*/
#include"CellList.cuh"
namespace CellList_ns{
  /*sortPos contains the particle position sorted in a way that ensures that all particles in a cell are one after the other*/
  /*You can identify where each cell starts and end by transversing this vector and searching for when the cell of a particle is 
    different from the previous one*/
  __global__ void fillCellListD(const real4 __restrict__ *sortPos,
				uint *cellStart, uint *cellEnd,
				uint N, NeighbourList::Utils utils){
    /*A thread per particle*/
    uint i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i<N){//If my particle is in range
      uint icell, icell2;
      /*Get my icell*/
      icell = utils.getCellIndex(utils.getCell(make_real3(sortPos[i])));
      
      /*Get the previous part.'s icell*/
      if(i>0){ /*Shared memory target VVV*/
	icell2 = utils.getCellIndex(utils.getCell(make_real3(sortPos[i-1])));
      }
      else
	icell2 = 0;
      //If my particle is the first or is in a different cell than the previous
      //my i is the start of a cell
      if(i ==0 || icell != icell2){
	//Then my particle is the first of my cell
	cellStart[icell] = i;
	
	//If my i is the start of a cell, it is also the end of the previous
	//Except if I am the first one
	if(i>0)
	  cellEnd[icell2] = i;
      }
      //If I am the last particle my cell ends 
      if(i == N-1) cellEnd[icell] = N;      
    }

  }
  
}

/*Construct the neighbour list*/
void CellList::makeNeighbourList(){
  /*The list only has to be recomputed one time per step*/
  if(last_step_updated == current_step) return;
  
  last_step_updated = current_step;
  /*Clear the cellStart array, if cellStart[icell] = 0xffFFffFFff, no particle is in that cell*/
  uint ncells = utils.cellDim.x*utils.cellDim.y*utils.cellDim.z;
  cudaMemset(cellStart.d_m, 0xffffffff, ncells*sizeof(uint));
  /*Order positions, grouping them by cell index*/
  this->reorderParticles();
  /*I need the conversion between old and new order,
    the reference to it changes each iteration due to the sorting process*/
  nl.particleIndex = particleIndex.d_m;

  /*Configure and launch the kernel to find cellStart and cellEnd*/
  int nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int nblocks  =  N/nthreads +  ((N%nthreads!=0)?1:0);     
  CellList_ns::fillCellListD<<<nblocks, nthreads>>>(sortPos,
						    cellStart, cellEnd,
						    N, utils);
}


/*Initialize a CellList instance*/
CellList::CellList(real rcut, real3 L, int N): NeighbourList_Base(rcut, L, N,true){
  /*Base initialization created some variables*/
  uint ncells = utils.cellDim.x*utils.cellDim.y*utils.cellDim.z;
  /*Initialize arrays*/
  cellStart = GPUVector<uint>(ncells);
  cellEnd = GPUVector<uint>(ncells);
  /*Fill the TransverseInfo*/
  nl.utils = utils;
  nl.texCellStart = cellStart.getTexture();
  nl.texCellEnd = cellEnd.getTexture();
  nl.texSortPos = sortPos.getTexture();
  /*No need for a texture, particleIndex is accesed in order*/
  nl.particleIndex = particleIndex.d_m;
  nl.N = N;
}


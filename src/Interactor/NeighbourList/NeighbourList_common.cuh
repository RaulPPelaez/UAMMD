/*Raul P. Pelaez 2017. Utils for neighbour lists*/
/*WARNING!: Device function header, only inline __device__ functions and struct definitions here!*/

/*
CONTAINS:
  -struct Utils
        Provides core information to order by cell hash neighbour lists. Such as parameters about the cells and functions to compute cell of a particle and 1D index of a cell coordinate
*/

#ifndef NEIGHBOURLIST_COMMON_CUH
#define NEIGHBOURLIST_COMMON_CUH
#include"utils/GPUutils.cuh"
namespace NeighbourList{

  struct Utils{
    /*A magic vector that transforms cell coordinates to 1D index when dotted*/
    /*Simply: 1, ncellsx, ncellsx*ncellsy*/
    int3 gridPos2CellIndex;
    real3 Lhalf; /*0.5*L*/
    int3 cellDim; //ncells in each size
    real3 cellSize;
    real3 invCellSize; /*The inverse of the cell size in each direction*/
    BoxUtils box;
    //Get linear index of a 3D cell, from 0 to ncells-1
    inline __device__ uint getCellIndex(int3 gridPos) const{
      return dot(gridPos, gridPos2CellIndex);
    }

    inline __device__ int3 getCell(real3 r) const{
      box.apply_pbc(r); //Reduce to MIC
      // return  int( (p+0.5L)/cellSize )
      int3 cell = make_int3((r+Lhalf)*invCellSize);
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

    //Apply pbc to a cell coordinates
    inline __device__ void pbc_cell(int3 &cell) const{
      if(cell.x<=-1) cell.x += cellDim.x;
      else if(cell.x>=cellDim.x) cell.x -= cellDim.x;
      
      if(cell.y<=-1) cell.y += cellDim.y;
      else if(cell.y>=cellDim.y) cell.y -= cellDim.y;
      
      if(cell.z<=-1) cell.z += cellDim.z;
      else if(cell.z>=cellDim.z) cell.z -= cellDim.z;
    }

    
  };


  
  
// }

// namespace Sorter{

  struct MortonHash{
    /*Interleave a 10 bit number in 32 bits, fill one bit and leave the other 2 as zeros.*/
    static inline __device__ uint encodeMorton(const uint &i){
      uint x = i;
      x &= 0x3ff;
      x = (x | x << 16) & 0x30000ff;
      x = (x | x << 8) & 0x300f00f;
      x = (x | x << 4) & 0x30c30c3;
      x = (x | x << 2) & 0x9249249;
      return x;
    }
    /*Fuse three 10 bit numbers in 32 bits, producing a Z order Morton hash*/
    static inline __device__ uint hash(const int3 &cell, const NeighbourList::Utils &utils){
      return encodeMorton(cell.x) | (encodeMorton(cell.y) << 1) | (encodeMorton(cell.z) << 2);
    }      
  };


  struct CellHash{
    static inline __device__ uint hash(const int3 &cell, const NeighbourList::Utils &utils){
      return utils.getCellIndex(cell);
    }
  };
  
  /*Assign a hash to each particle*/
  template<class HashComputer>
  __global__ void computeHash(real4 __restrict__ *pos,
			      uint  __restrict__ *particleIndex,
			      uint  __restrict__ *particleHash, int N,
			      NeighbourList::Utils utils){
    const uint i = blockIdx.x*blockDim.x + threadIdx.x;  
    if(i>=N) return;
    const real3 p = make_real3(pos[i]);

    const int3 cell = utils.getCell(p);
    /*The particleIndex array will be sorted by the hashes, any order will work*/
    const uint hash = HashComputer::hash(cell, utils);
    
    /*Before ordering by hash the index in the array is the index itself*/
    particleIndex[i] = i;
    particleHash[i]  = hash;
  }
  

}
#endif
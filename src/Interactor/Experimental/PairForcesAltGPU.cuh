/*Raul P. Pelaez 2016. Short range pair forces Interactor GPU callers and kernels.

Functions to compute the pair, short range, force acting on each particle.

Neighbour list GPU implementation using hash short with cell index as hash.


References:
http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

More info in the .cu
*/

#ifndef PAIRFORCESALTGPU_CUH
#define PAIRFORCESALTGPU_CUH


namespace pair_forces_alt_ns{
  //Stores some simulation parameters to upload as constant memory.
  struct Params{
    uint *errorFlag;
    float rmax,rmax2, rc, rc2, invrc2;
    float invCellSize, getCellFactor;
    int3 cellDim, gridPos2cellIndex;

    float L, invL;
  
  };
  //Stores some simulation parameters to upload as constant memory.
  void initPairForcesAltGPU(Params m_params, float4* pos, float4 *sortPos, uint *NBL, uint N, uint maxNPerCell);
  void updateParamsPairForcesAltGPU(Params m_params);
  bool checkBinningCells(float4 *old_pos,  float4 *cur_pos, uint N, float rthreshold);

  bool makeNeighbourListGPU2(uint *&cellIndex, uint *cellSize, uint *&particleIndex,
			     float rmax,
			     uint *CELL, uint ncells,
			     uint *NBL, uint *NNeigh,
			     float4 *pos, float4* old_pos, float4 *sortPos,
			     uint N, uint maxNPerCell);

  //void makeNeighbourList(uint *CELL, uint *NBL, uint *Nneigh, uint ncells, uint N, uint maxNPerCell);


  //void updateList(uint *CELL, uint *NBL, uint *Nneigh, float4 *pos, uint N, uint ncells, uint maxNPerCell);
  void computePairForce(uint *NBL, uint *Nneigh, uint *particleIndex,
			float4 *pos, float4 *sortPos, float4 *force,
			uint N, uint maxNPerCell);


  void rebindGPU(uint maxNPerCell, uint N,uint * NBL);
}
#endif









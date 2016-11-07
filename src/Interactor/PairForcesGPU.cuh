/*Raul P. Pelaez 2016. Short range pair forces Interactor GPU callers and kernels.

Functions to compute the pair, short range, force acting on each particle.

Neighbour list GPU implementation using hash short with cell index as hash.


References:
http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

More info in the .cu
*/

#ifndef PAIRFORCESGPU_CUH
#define PAIRFORCESGPU_CUH
#include"globals/defines.h"

namespace pair_forces_ns{
  
  //Stores some simulation parameters to upload as constant memory.
  struct Params{
    real3 cellSize, invCellSize;
    int ncells;
    int3 cellDim;
    real rcut, invrc, invrc2;
    int3 gridPos2CellIndex;
    
    cudaTextureObject_t texForce, texEnergy;
    cudaTextureObject_t texSortPos, texPos;
    cudaTextureObject_t texCellStart, texCellEnd;
    uint N;
    real3 L, invL;

    uint ntypes;
    real2 *potParams;
  };
  //Stores some simulation parameters to upload as constant memory, the rest are available in Params.
  struct ParamsDPD{
    real gamma, noiseAmp, A;
    cudaTextureObject_t texSortVel;
  };

  // void initGPU(Params &m_params,
  // 	       cudaTextureObject_t texForce, cudaTextureObject_t texEnergy,
  // 	       uint *cellStart, uint *cellEnd, uint* particleIndex, uint ncells,
  // 	       real4 *sortPos, real4 *pos, uint N);
  void initGPU(Params &m_params, uint N);  
  void initDPDGPU(ParamsDPD &m_params);


  void updateParams(Params m_params);

  void makeCellList(real4 *pos, real4 *sortPos,
		    uint *&particleIndex, uint *&particleHash,
		    uint *cellStart, uint *cellEnd,
		    uint N, uint ncells);
  
  void makeCellListDPD(real4 *pos, real3* vel,  real4 *sortPos, real4 *sortVel,
		       uint *&particleIndex, uint *&particleHash,
		       uint *cellStart, uint *cellEnd,
		       uint N, uint ncells);


  void computePairForce(real4 *sortPos, real4 *force,
			uint *cellStart, uint *cellEnd,
			uint *particleIndex, 
			uint N);

  void computePairForceDPD(real4 *force,
			   uint *particleIndex,
			   uint *cellStart,
			   uint *cellEnd,
			   uint N, unsigned long long int seed);



  real computePairEnergy(real4 *sortPos, real *energy,		  
			  uint *cellStart, uint *cellEnd,
			  uint *particleIndex, 
			  uint N);
  real computePairVirial(real4 *sortPos, real *virial,		  
			  uint *cellStart, uint *cellEnd,
			  uint *particleIndex, 
			  uint N);

}
#endif









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
#include"utils/Texture.h"

namespace pair_forces_ns{
  
  //Stores some simulation parameters to upload as constant memory.
  struct Params{
    real3 cellSize, invCellSize;
    int ncells;
    int3 cellDim;
    real rcut_pot, invrc_pot, invrc2_pot;
    real rcut;
    int3 gridPos2CellIndex;
    
    TexReference texForce, texEnergy;
    TexReference texSortPos, texPos;
    TexReference texCellStart, texCellEnd;
    uint N;
    real3 L, invL;

    uint ntypes;
    real2 *potParams;
  };
  //Stores some simulation parameters to upload as constant memory, the rest are available in Params.
  struct ParamsDPD{
    real gamma, noiseAmp, A;
    TexReference texSortVel;
  };

  // void initGPU(Params &m_params,
  // 	       cudaTextureObject_t texForce, cudaTextureObject_t texEnergy,
  // 	       uint *cellStart, uint *cellEnd, uint* particleIndex, uint ncells,
  // 	       real4 *sortPos, real4 *pos, uint N);
  void initGPU(Params &m_params, Params *&d_params, uint N, size_t potSize);  
  void initDPDGPU(ParamsDPD &m_params);


  void updateParams(Params m_params);
  void updateParamsFromGPU(Params *d_params);

  void makeCellList(real4 *pos, real4 *sortPos,// real4 *old_pos,
		    uint *&particleIndex, uint *&particleHash,
		    uint *cellStart, uint *cellEnd,
		    uint N, uint ncells);
  bool needsUpdateGPU(real4 *pos, real4 *old_pos, real threshold, uint N);
  
  void makeCellListDPD(real4 *pos, real3* vel,  real4 *sortPos, real4 *sortVel,
		       uint *&particleIndex, uint *&particleHash,
		       uint *cellStart, uint *cellEnd,
		       uint N, uint ncells);


  template<class Transverser>
  void computeWithListGPU(Transverser t,
			  uint *cellStart, uint *cellEnd,
			  uint *particleIndex, 
			  uint N);

  
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

  void reorderPosGPU( real4 *sortPos, real4 *pos, uint *particleIndex, uint N);

}
#endif









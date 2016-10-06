/*Raul P. Pelaez 2016. Short range pair forces Interactor GPU callers and kernels.

Functions to compute the pair, short range, force acting on each particle.

Neighbour list GPU implementation using hash short with cell index as hash.


References:
http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

More info in the .cu
*/

#ifndef PAIRFORCESGPU_CUH
#define PAIRFORCESGPU_CUH

namespace pair_forces_ns{
  //Stores some simulation parameters to upload as constant memory.
  struct Params{
    float cellSize, invCellSize;
    int ncells;
    int3 cellDim;
    float L, invL, rcut, invrc, invrc2;
    float3 getCellFactor;
    int3 gridPos2CellIndex;
    cudaTextureObject_t texForce, texEnergy;
  };
  //Stores some simulation parameters to upload as constant memory, the rest are available in Params.
  struct ParamsDPD{
    float gamma, noiseAmp, A;
  };

  void initPairForcesGPU(Params &m_params,
			 cudaTextureObject_t texForce, cudaTextureObject_t texEnergy,
			 uint *cellStart, uint *cellEnd, uint* particleIndex, uint ncells,
			 float4 *sortPos, float4 *pos, uint N);
  
  void initPairForcesDPDGPU(ParamsDPD &m_params, float4* sortVel, uint N);


  void updateParams(Params m_params);

  void makeCellList(float4 *pos, float4 *sortPos,
		    uint *&particleIndex, uint *&particleHash,
		    uint *cellStart, uint *cellEnd,
		    uint N, uint ncells);
  
  void makeCellListDPD(float4 *pos, float3* vel,  float4 *sortPos, float4 *sortVel,
		       uint *&particleIndex, uint *&particleHash,
		       uint *cellStart, uint *cellEnd,
		       uint N, uint ncells);


  void computePairForce(float4 *sortPos, float4 *force,
			uint *cellStart, uint *cellEnd,
			uint *particleIndex, 
			uint N);

  void computePairForceDPD(float4 *force,
			   uint *particleIndex,
			   uint N, unsigned long long int seed);



  float computePairEnergy(float4 *sortPos, float *energy,		  
			  uint *cellStart, uint *cellEnd,
			  uint *particleIndex, 
			  uint N);
  float computePairVirial(float4 *sortPos, float *virial,		  
			  uint *cellStart, uint *cellEnd,
			  uint *particleIndex, 
			  uint N);

}
#endif









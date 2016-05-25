/*
Raul P. Pelaez 2016. Interactor GPU kernels and callers.

Functions to compute the force acting on each particle and integrate movement

Neighbour list GPU implementation using hash short with cell index as hash.



References:
http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

More info in the .cu
*/

#ifndef INTERACTORGPU_CUH
#define INTERACTORGPU_CUH

//Stores some simulation parameters to upload as constant memory.
struct Params{
  float cellSize, invCellSize;
  int ncells;
  int ycells, xcells, zcells;  
  float L, invL, rcut, invrc2;
};

void initGPU(Params m_params, float *potDevPtr, size_t potSize,
	     uint *cellStart, uint *cellEnd, uint* particleIndex, uint ncells,
	     float *sortPos, uint N);

void integrate(float *pos, float *vel, float *force, float dt, uint N, int step, bool dump=false);

void calcCellIndex(float *pos, uint *cellIndex, uint *particleIndex, uint N);

void sortCellIndex(uint *&cellIndex, uint *&particleIndex, uint N);

void reorderAndFind(float *sortPos,
		    uint *cellIndex, uint *particleIndex, 
		    uint *cellStart, uint *cellEnd, uint ncells,
		    float*pos, uint N);

void computeForce(float *sortPos, float *force,
		  uint *cellStart, uint *cellEnd,
		  uint *particleIndex, 
		  uint N);


#endif









/*Raul P. Pelaez 2016. Short range pair forces Interactor GPU callers and kernels.

Functions to compute the force acting on each particle

Neighbour list GPU implementation using hash short with cell index as hash.

References:
http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

TODO:
100- Use Z-order curve as hash instead of cell index to improve memory coherence when traveling the neighbour cells
90- Add support for particle types, encode in pos.w
90- Make energy measure custom for each potential, currently only LJ, hardcoded.
50- Try bindless textures again.
40- pbc_cells could be done better, this could improve force compute
10- There is no need to reconstruct the neighbour list from scratch each step,
  although computing the force is 50 times as expensive as this right now.
10- Find a way to properly handle the alternate arrays in sortCellIntex
*/

#include<cub/cub/cub.cuh>
#include<curand_kernel.h>
#include"PairForcesGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"

#include<thrust/device_ptr.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/sort.h>
#include<iostream>


typedef unsigned long long int ullint;

#define BLOCKSIZE 128

using namespace thrust;
using std::cerr;
using std::endl;

__constant__ PairForcesParams params; //Simulation parameters in constant memory, super fast access
__constant__ PairForcesParamsDPD paramsDPD; //Simulation parameters in constant memory, super fast access

//Texture references for scattered access
texture<uint> texCellStart, texCellEnd, texParticleIndex;
texture<float4> texSortPos;
texture<float4> texSortVel; //For DPD
texture<float, 1, cudaReadModeElementType> texForce; cudaArray *dF;
//texture<float,1 , cudaReadModeElementType> texEnergy; cudaArray *dE;


uint GPU_Nblocks;
uint GPU_Nthreads;


//Initialize gpu variables 
void initPairForcesGPU(PairForcesParams m_params,
		       float *potForceData, float *potEnergyData, size_t potSize,
		       uint *cellStart, uint *cellEnd, uint* particleIndex, uint ncells,
		       float4 *sortPos, uint N){

  /*Precompute some inverses to save time later*/
  m_params.invrc2 = 1.0f/(m_params.rcut*m_params.rcut);
  m_params.invrc = 1.0f/(m_params.rcut);
  m_params.invL = 1.0f/m_params.L;
  m_params.invCellSize = 1.0f/m_params.cellSize;
  m_params.getCellFactor = 0.5f*m_params.L*m_params.invCellSize;
  
  /*Texture bindings, these ones are accessed by element*/ 
  gpuErrchk(cudaBindTexture(NULL, texCellStart, cellStart, ncells*sizeof(uint)));
  gpuErrchk(cudaBindTexture(NULL, texCellEnd,   cellEnd,   ncells*sizeof(uint)));
  gpuErrchk(cudaBindTexture(NULL, texParticleIndex,   particleIndex,   (N+1)*sizeof(uint)));
  gpuErrchk(cudaBindTexture(NULL, texSortPos, sortPos, N*sizeof(float4)));

  /*Create and bind force texture, this needs interpolation*/
  cudaChannelFormatDesc channelDesc;
  channelDesc = cudaCreateChannelDesc(32, 0,0,0, cudaChannelFormatKindFloat);

  gpuErrchk(cudaMallocArray(&dF,
			    &channelDesc,
			    potSize/sizeof(float),1));

  gpuErrchk(cudaMemcpyToArray(dF, 0,0, potForceData, potSize, cudaMemcpyHostToDevice));

  texForce.normalized = true; //The values are fetched between 0 and 1
  texForce.addressMode[0] = cudaAddressModeClamp; //0 outside [0,1]
  texForce.filterMode = cudaFilterModeLinear; //Linear filtering

  /*Texture binding*/
  gpuErrchk(cudaBindTextureToArray(texForce, dF, channelDesc));
  
  /**SAME WITH THE ENERGY**/
  // gpuErrchk(cudaMallocArray(&dE,
  // 			    &channelDesc,
  // 			    potSize/sizeof(float),1));
  // gpuErrchk(cudaMemcpyToArray(dE, 0,0, potEnergyData, potSize, cudaMemcpyHostToDevice));
  // /*Texture binding*/
  // gpuErrchk(cudaBindTextureToArray(texEnergy, dE, channelDesc));


  /*Upload parameters to constant memory*/
  gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(PairForcesParams)));



  /*Each particle is asigned a thread*/
  GPU_Nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  GPU_Nblocks  =  N/GPU_Nthreads +  ((N%GPU_Nthreads!=0)?1:0); 
}



void initPairForcesDPDGPU(PairForcesParamsDPD m_params, float4* sortVel, uint N){

  gpuErrchk(cudaMemcpyToSymbol(paramsDPD, &m_params, sizeof(PairForcesParamsDPD)));


  gpuErrchk(cudaBindTexture(NULL, texSortVel, sortVel, N*sizeof(float4)));
}

/****************************HELPER FUNCTIONS*****************************************/
//MIC algorithm
inline __device__ void apply_pbc(float3 &r){
  r -= floorf(r*params.invL+0.5f)*params.L; 
}
inline __device__ void apply_pbc(float4 &r){
  r -= floorf(r*params.invL+0.5f)*params.L; //MIC algorithm
}

//Get the 3D cell p is in, just pos in [0,L] divided by ncells(vector) .INT DANGER.
inline __device__ int3 getCell(float3 p){
  apply_pbc(p); //Reduce to MIC
  // return  int( (p+0.5L)/cellSize )
  return make_int3( p*params.invCellSize + params.getCellFactor ); 
}
inline __device__ int3 getCell(float4 p){
  apply_pbc(p); //Reduce to MIC
  // return  int( (p+0.5L)/cellSize )
  return make_int3( p*params.invCellSize + params.getCellFactor ); 
}

//Apply pbc to a cell coordinates
inline __device__ void pbc_cell(int3 &cell){
  if(cell.x==-1) cell.x = params.xcells-1;
  else if(cell.x==params.xcells) cell.x = 0;

  if(cell.y==-1) cell.y = params.ycells-1;
  else if(cell.y==params.ycells) cell.y = 0;

  if(cell.z==-1) cell.z = params.zcells-1;
  else if(cell.z==params.zcells) cell.z = 0;
}
//Get linear index of a 3D cell, from 0 to ncells-1
inline __device__ uint getCellIndex(int3 gridPos){
  return gridPos.x
    +gridPos.y*params.xcells
    +gridPos.z*params.xcells*params.ycells;
}

/****************************************************************************************/


//Compute the icell of each particle
__global__ void calcCellIndexD(uint *cellIndex, uint *particleIndex, 
			       const float4 __restrict__ *pos, uint N){
  uint index = blockIdx.x*blockDim.x + threadIdx.x;  
  if(index>N) return;
  float4 p = pos[index];

  int3 gridPos = getCell(p);
  int icell = getCellIndex(gridPos);
  /*Before ordering by icell the index in the array is the index!*/
  particleIndex[index] = index;
  cellIndex[index]  = icell;
  
}  
//CPU kernel caller
void calcCellIndex(float4 *pos, uint *cellIndex, uint *particleIndex, uint N){
  calcCellIndexD<<<GPU_Nblocks, GPU_Nthreads>>>(cellIndex, particleIndex, pos, N);
  //cudaCheckErrors("Calc hash");					   
}



//Sort the particleIndex list by cell index,
// this allows to access the neighbour list of each particle fast and coalesced
void sortCellIndex(uint *&cellIndex, uint *&particleIndex, uint N){
  //This uses the CUB API to perform a radix sort
  //CUB orders by key an array pair and copies them onto another pair
  //This function stores an internal key/value pair and switches the arrays each time
   static bool init = false;
   static void *d_temp_storage = NULL;
   static size_t temp_storage_bytes = 0; //Additional storage needed by cub
   static uint *cellIndex_alt = NULL, *particleIndex_alt = NULL; //Additional key/value pair

   static cub::DoubleBuffer<uint> d_keys;
   static cub::DoubleBuffer<uint> d_values;
   /**Initialize CUB at first call**/
   if(!init){
     /*Allocate temporal value/key pair*/
     gpuErrchk(cudaMalloc(&cellIndex_alt, N*sizeof(uint)));
     gpuErrchk(cudaMalloc(&particleIndex_alt, N*sizeof(uint)));
     
     /*Create this CUB like data structure*/
     d_keys = cub::DoubleBuffer<uint>(cellIndex, cellIndex_alt);    
     d_values = cub::DoubleBuffer<uint>(particleIndex, particleIndex_alt);
     /*On first call, this function only computes the size of the required temporal storage*/
     cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
   				    d_keys, 
   				    d_values, N);
     /*Allocate temporary storage*/
     gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
     init = true;
   }

   /**Perform the Radix sort on the index/cell pair**/
   cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				   d_keys, 
				   d_values, N); 
   /**Switch the references**/
   cellIndex     = d_keys.Current();
   particleIndex = d_values.Current();

   cellIndex_alt     = d_keys.Alternate();
   particleIndex_alt = d_values.Alternate();

   /*Very important, fix the texture reference!!*/
   gpuErrchk(cudaBindTexture(NULL, texParticleIndex,   particleIndex,   (N+1)*sizeof(uint)));

   //Thrust is slower and more memory hungry, for it is a higher level call
   // thrust::sort_by_key(device_ptr<uint>(cellIndex),
   // 		      device_ptr<uint>(cellIndex+N),
   // 		      device_ptr<uint>(particleIndex));

   //cudaCheckErrors("Sort hash");					   
}

//Create CellStart and CellEnd, copy pos onto sortPos
__global__ void reorderAndFindD(float4 *sortPos,
				uint *cellIndex, uint *particleIndex, 
				uint *cellStart, uint *cellEnd,
				float4 *pos,
				uint N){
  uint index = blockIdx.x*blockDim.x + threadIdx.x;
  uint icell, icell2;

  if(index<N){//If my particle is in range
    icell = cellIndex[index]; //Get my icell
    if(index>0)icell2 = cellIndex[index-1];//Get the previous part.'s icell
    else icell2 = 0;
    //If my particle is the first or is in a different cell than the previous
    //my index is the start of a cell
    if(index ==0 || icell != icell2){
      //Then my particle is the first of my cell
      cellStart[icell] = index;
      //If my index is the start of a cell, it is also the end of the previous
      //Except if I am the first one
      if(index>0)
	cellEnd[icell2] = index;
    }
    //If I am the last particle my cell ends 
    if(index == N-1) cellEnd[icell] = index+1;

    //Copy pos into sortPos
    //uint sortIndex   = particleIndex[index];
    uint sortIndex   = tex1Dfetch(texParticleIndex, index);
    sortPos[index]   = pos[sortIndex];
  }

}

//CPU kernel caller
void reorderAndFind(float4 *sortPos,
		    uint *cellIndex, uint *particleIndex, 
		    uint *cellStart, uint *cellEnd, uint ncells,
		    float4 *pos, uint N){
  //Reset CellStart
  cudaMemset(cellStart, 0xffffffff, ncells*sizeof(uint));
  //CellEnd does not need reset, a cell with cellStart=0xffffff is not checked for a cellEnd
  reorderAndFindD<<<GPU_Nblocks, GPU_Nthreads>>>(sortPos,
						 cellIndex, particleIndex,
						 cellStart, cellEnd,
						 pos, N);
  //cudaCheckErrors("Reorder and find");					   
}



/***************************************FORCE*****************************/


//Computes the force between to positions
inline __device__ float4 forceij(const float4 &R1,const float4 &R2){

  float3 r12 = make_float3(R2-R1);

  apply_pbc(r12);

  /*Squared distance between 0 and 1*/
  float r2 = dot(r12,r12);
  float r2c = r2*params.invrc2;
  /*Check if i==j. This way reduces warp divergence and its faster than checking i==j outside*/
  //  if(r2c==0.0f) return make_float4(0.0f);  //Both cases handled in texForce
  /*Beyond rcut..*/
  //else if(r2c>=1.0f) return make_float4(0.0f);
  /*Get the force from the texture*/
  float fmod = tex1D(texForce, r2c);
   // float invr2 = 1.0f/r2;
   //  float invr6 = invr2*invr2*invr2;
    //  float invr8 = invr6*invr2;
    //float E =  2.0f*invr6*(invr6-1.0f);
    //float fmod = -48.0f*invr8*invr6+24.0f*invr8;
  return make_float4(fmod*r12);
 }

//Computes the force acting on particle index from particles in cell cell
__device__ float4 forceCell(const int3 &cell, const uint &index,
			    const float4 &pos){
  uint icell  = getCellIndex(cell);
  /*Index of the first particle in the cell's list*/ 
  uint firstParticle = tex1Dfetch(texCellStart, icell);

  float4 force = make_float4(0.0f);
  float4 posj;

  /*Index of the last particle in the cell's list*/
  uint lastParticle = tex1Dfetch(texCellEnd, icell);
  /*Because the list is ordered, all the particle indices in the cell are coalescent!*/
  /*If there are no particles in the cell, firstParticle=0xffffffff, the loop is not computed*/
  /*The fetch of lastParticle eitherway reduces branch divergency and is actually faster than checking
    firstParticle before fetching*/
  for(uint j=firstParticle; j<lastParticle; j++){
    /*Retrieve j pos*/
    posj = tex1Dfetch(texSortPos, j);
    /*Add force, i==j is handled in forceij */
    force += forceij(pos, posj);
  }
   
  return force;
}


//Kernel to compute the force acting on all particles
__global__ void computeForceD(float4* __restrict__ newForce,
			      const uint* __restrict__ particleIndex, 
			      uint N){
  /*Travel the particles per sort order*/
  uint ii =  blockIdx.x*blockDim.x + threadIdx.x;
  
  //Grid-strid loop
  for(int index = ii; index<N; index += blockDim.x * gridDim.x){
    /*Compute force acting on particle particleIndex[index], index in the new order*/
    float4 pos = tex1Dfetch(texSortPos, index);
  
    float4 force = make_float4(0.0f);
    int3 celli = getCell(pos);

    int x,y,z;
    int3 cellj;
    /**Go through all neighbour cells**/
    //For some reason unroll doesnt help here
    for(z=-1; z<=1; z++)
      for(y=-1; y<=1; y++)
	for(x=-1; x<=1; x++){
	  cellj = celli+make_int3(x,y,z);
	  pbc_cell(cellj);	
	  force += forceCell(cellj, index, pos);
	}
    /*Write force with the original order*/
    uint pi = tex1Dfetch(texParticleIndex, index); 
    newForce[pi] += force;
   }
}
__global__ void computeForceDnaive(float4* __restrict__ newForce,
				   const uint* __restrict__ particleIndex, 
				   uint N){
  /*Travel the particles per sort order*/
  uint index =  blockIdx.x*blockDim.x + threadIdx.x;
  if(index>=N) return;
  
  /*Compute force acting on particle particleIndex[index], index in the new order*/
  float4 pos = tex1Dfetch(texSortPos, index);
  float4 posj;
  float4 force = make_float4(0.0f);
  for(int i=0; i<N; i++){
    posj = tex1Dfetch(texSortPos, i);
    force += forceij(pos, posj);
  }

  /*Write force with the original order*/
  uint pi = tex1Dfetch(texParticleIndex, index); 
  newForce[pi] += force;
}

//CPU kernel caller
void computePairForce(float4 *sortPos, float4 *force,
		  uint *cellStart, uint *cellEnd,
		  uint *particleIndex, 
		  uint N){
  computeForceD<<<GPU_Nblocks, GPU_Nthreads>>>(force,
					       particleIndex,
					       N);
  //cudaCheckErrors("computeForce");
}







/****************************ENERGY***************************************/


//Computes the energy between to positions, no cutoff
inline __device__ float energyij(const float4 &R1,const float4 &R2){

  float3 r12 = make_float3(R2-R1);

  apply_pbc(r12);

  float r2 = dot(r12,r12);
  /*Squared distance between 0 and 1*/
  //float r2c = r2*params.invrc2;
  /*Check if i==j. This way reduces warp divergence and its faster than checking i==j outside*/
  if(r2==0.0f) return 0.0f;
  else if(r2*params.invrc2>=1.0f) return 0.0f;
  float invr2 = 1.0f/r2;
  float invr6 = invr2*invr2*invr2;
  float E =  2.0f*invr6*(invr6-1.0f);
  return E;
 }


//Computes the energy acting on particle index from particles in cell cell
__device__ float energyCell(const int3 &cell, const uint &index,
			   const float4 &pos){
  uint icell  = getCellIndex(cell);
  /*Index of the first particle in the cell's list*/ 
  uint firstParticle = tex1Dfetch(texCellStart, icell);

  float energy = 0.0f;
  float4 posj;
  /*Exact copy of forceCell*/
  uint lastParticle = tex1Dfetch(texCellEnd, icell);
  for(uint j=firstParticle; j<lastParticle; j++){
    posj = tex1Dfetch(texSortPos, j);
    energy += energyij(pos, posj);
  }
  return energy;
}




//Kernel to compute the force acting on all particles
__global__ void computeEnergyDnaive(float* __restrict__ Energy,
				    const uint* __restrict__ particleIndex, 
				    uint N){
  /*Travel the particles per sort order*/
  uint index =  blockIdx.x*blockDim.x + threadIdx.x;
  if(index>=N) return;
  
  /*Compute force acting on particle particleIndex[index], index in the new order*/
  float4 pos = tex1Dfetch(texSortPos, index), posj;
  
  float energy = 0.0f;
  //  int3 celli = getCell(pos);

  for(int j=0; j<N; j++){
    posj = tex1Dfetch(texSortPos, j);
    energy += energyij(pos, posj);
  }

  /*Write force with the original order*/
  uint pi = tex1Dfetch(texParticleIndex, index); 
  Energy[pi] = energy;
}
//Kernel to compute the force acting on all particles
__global__ void computeEnergyD(float* __restrict__ Energy,
				    const uint* __restrict__ particleIndex, 
				    uint N){
  /*Travel the particles per sort order*/
  uint index =  blockIdx.x*blockDim.x + threadIdx.x;
  if(index>=N) return;
  
  /*Compute force acting on particle particleIndex[index], index in the new order*/
  float4 pos = tex1Dfetch(texSortPos, index);
  
  float energy = 0.0f;
  int3 celli = getCell(pos);
  int x,y,z;
  int3 cellj;
  /**Go through all neighbour cells**/
  for(z=-1; z<=1; z++)
    for(y=-1; y<=1; y++)
      for(x=-1; x<=1; x++){
	cellj = celli+make_int3(x,y,z);
	pbc_cell(cellj);	
	energy += energyCell(cellj, index, pos);
      }

  /*Write force with the original order*/
  uint pi = tex1Dfetch(texParticleIndex, index); 
  Energy[pi] = energy;
}

//CPU kernel caller
float computePairEnergy(float4 *sortPos, float *energy,
		  uint *cellStart, uint *cellEnd,
		  uint *particleIndex, 
		  uint N){
  computeEnergyD<<<GPU_Nblocks, GPU_Nthreads>>>(energy,
						particleIndex,
   						N);

  device_ptr<float> d_e(energy);
  float sum;
  sum = thrust::reduce(d_e, d_e+N, 0.0f);
  return (sum/(float)N);

  //cudaCheckErrors("computeForce");
}


/****************************VIRIAL***************************************/


//Computes the virial between to positions
inline __device__ float virialij(const float4 &R1,const float4 &R2){

  float3 r12 = make_float3(R2-R1);

  apply_pbc(r12);

  /*Squared distance between 0 and 1*/
  float r2 = dot(r12,r12);
  float r2c = r2*params.invrc2;
  //if(r2c==0.0f) return 0.0f; //No need to check i==j, tex1D(texForce, 0.0) = 0.0
  /*Beyond rcut..*/
  //if(r2c>=1.0f) return 0.0f; //Also 0 in texForce
  /*Get the force from the texture*/
  float fmod = tex1D(texForce, r2c);
  // P = rhoKT + (1/2dV)sum_ij( Fij·rij )
  return dot(fmod*r12,r12);
 }



//Computes the virial acting on particle index from particles in cell cell
__device__ float virialCell(const int3 &cell, const uint &index,
			   const float4 &pos){
  uint icell  = getCellIndex(cell);
  /*Index of the first particle in the cell's list*/ 
  uint firstParticle = tex1Dfetch(texCellStart, icell);

  float virial = 0.0f;
  float4 posj;

  /*Exact copy of forceCell*/
  uint lastParticle = tex1Dfetch(texCellEnd, icell);
  for(uint j=firstParticle; j<lastParticle; j++){
    posj = tex1Dfetch(texSortPos, j);
    virial += virialij(pos, posj);
    }
   
  return virial;
}


//Kernel to compute the force acting on all particles
__global__ void computeVirialDnaive(float* __restrict__ Virial,
				    const uint* __restrict__ particleIndex, 
				    uint N){
  /*Travel the particles per sort order*/
  uint index =  blockIdx.x*blockDim.x + threadIdx.x;
  if(index>=N) return;
  
  /*Compute force acting on particle particleIndex[index], index in the new order*/
  float4 pos = tex1Dfetch(texSortPos, index), posj;
  
  float virial = 0.0f;
  //  int3 celli = getCell(pos);

  for(int j=0; j<N; j++){
    posj = tex1Dfetch(texSortPos, j);
    virial += virialij(pos, posj);
  }

  /*Write force with the original order*/
  uint pi = tex1Dfetch(texParticleIndex, index); 
  Virial[pi] = virial;
}
//Kernel to compute the force acting on all particles
__global__ void computeVirialD(float* __restrict__ Virial,
				    const uint* __restrict__ particleIndex, 
				    uint N){
  /*Travel the particles per sort order*/
  uint index =  blockIdx.x*blockDim.x + threadIdx.x;
  if(index>=N) return;
  
  /*Compute force acting on particle particleIndex[index], index in the new order*/
  float4 pos = tex1Dfetch(texSortPos, index);
  
  float virial = 0.0f;
  int3 celli = getCell(pos);
  int x,y,z;
  int3 cellj;
  /**Go through all neighbour cells**/
  for(z=-1; z<=1; z++)
    for(y=-1; y<=1; y++)
      for(x=-1; x<=1; x++){
	cellj = celli+make_int3(x,y,z);
	pbc_cell(cellj);	
	virial += virialCell(cellj, index, pos);
      }

  /*Write force with the original order*/
  uint pi = tex1Dfetch(texParticleIndex, index); 
  Virial[pi] = virial;
}

//CPU kernel caller
float computePairVirial(float4 *sortPos, float *virial,
		  uint *cellStart, uint *cellEnd,
		  uint *particleIndex, 
		  uint N){
  computeVirialD<<<GPU_Nblocks, GPU_Nthreads>>>(virial,
						particleIndex,
   						N);

  device_ptr<float> d_vir(virial);
  float sum;
  sum = thrust::reduce(d_vir, d_vir+N, 0.0f);
  return (sum/2.0f);

  //cudaCheckErrors("computeForce");
}



/*******************************************DPD********************************************/

/**********************FORCE********************/


//Create CellStart and CellEnd, copy pos onto sortPos
__global__ void reorderAndFindDDPD(float4 *sortPos, float4 *sortVel,
				   uint *cellIndex, uint *particleIndex, 
				   uint *cellStart, uint *cellEnd,
				   const float4 *pos, const float3 *vel,
				   uint N){
  uint index = blockIdx.x*blockDim.x + threadIdx.x;
  uint icell, icell2;

  if(index<N){//If my particle is in range
    icell = cellIndex[index]; //Get my icell
    if(index>0)icell2 = cellIndex[index-1];//Get the previous part.'s icell
    else icell2 = 0;
    //If my particle is the first or is in a different cell than the previous
    //my index is the start of a cell
    if(index ==0 || icell != icell2){
      //Then my particle is the first of my cell
      cellStart[icell] = index;
      //If my index is the start of a cell, it is also the end of the previous
      //Except if I am the first one
      if(index>0)
	cellEnd[icell2] = index;
    }
    //If I am the last particle my cell ends 
    if(index == N-1) cellEnd[icell] = index+1;

    //Copy pos into sortPos
    //uint sortIndex   = particleIndex[index];
    uint sortIndex   = tex1Dfetch(texParticleIndex, index);
    sortPos[index]   = pos[sortIndex];
    sortVel[index]   = make_float4(vel[sortIndex], 0.0f);
  }

}
//CPU kernel caller
void reorderAndFindDPD(float4 *sortPos, float4* sortVel,
		       uint *cellIndex, uint *particleIndex, 
		       uint *cellStart, uint *cellEnd, uint ncells,
		       float4 *pos, float3* vel, uint N){
  //Reset CellStart
  cudaMemset(cellStart, 0xffffffff, ncells*sizeof(uint));
  //CellEnd does not need reset, a cell with cellStart=0xffffff is not checked for a cellEnd
  reorderAndFindDDPD<<<GPU_Nblocks, GPU_Nthreads>>>(sortPos, sortVel,
						    cellIndex, particleIndex,
						    cellStart, cellEnd,
						    pos, vel, N);
  //cudaCheckErrors("Reorder and find");					   
}



inline __device__ float randGPU(const ullint &seed, curandState *rng){
  curand_init(seed, 0, 0, rng);
  return curand_normal(rng);
}


//Computes the force between to positions
inline __device__ float4 forceijDPD(const float4 &R1,const float4 &R2,
				    const float4 &V1,const float4 &V2, const float &randij){
  
  float3 r12 = make_float3(R1-R2);
  float3 v12 = make_float3(V1-V2);
  
  apply_pbc(r12);

  float r2 = dot(r12,r12);
  /*Squared distance between 0 and 1*/
  float r2c = r2*params.invrc2;
  
  float fmod= 0.0f;
  
  float w = 0.0f; //The intensity of the DPD thermostat 
  float rinv = 0.0f;
  if(r2c<1.0f){
    if(r2c==0.0f) return make_float4(0.0f);
    //w = r-rc -> linear
    rinv = rsqrt(r2);
    w = rinv-params.invrc;
  }
  else return make_float4(0.0f);
  //fmod = paramsDPD.A*w; //Soft force
  
  fmod -= tex1D(texForce, r2c); //Conservative force
  fmod -= paramsDPD.gamma*w*w*dot(r12,v12); //Damping
  fmod += paramsDPD.noiseAmp*randij*w; //Random force
  return make_float4(fmod*r12);
}

//Computes the force acting on particle index from particles in cell cell
__device__ float4 forceCellDPD(const int3 &cell, const uint &index,
			       const float4 &pos,
			       const float4 &veli,
			       uint N,
			       curandState &rng, const ullint &seed){
  uint icell  = getCellIndex(cell);
  /*Index of the first particle in the cell's list*/ 
  uint firstParticle = tex1Dfetch(texCellStart, icell);

  float4 force = make_float4(0.0f);
  float4 posj, velj;
  
  /*Index of the last particle in the cell's list*/
  uint lastParticle = tex1Dfetch(texCellEnd, icell);
  /*Because the list is ordered, all the particle indices in the cell are coalescent!*/
  /*If there are no particles in the cell, firstParticle=0xffffffff, the loop is not computed*/
  /*The fetch of lastParticle eitherway reduces branch divergency and is actually faster than checking
    firstParticle before fetching*/
  float randij;
  ullint i0, j0;
  for(uint j=firstParticle; j<lastParticle; j++){
    /*Retrieve j pos and vel*/
    posj = tex1Dfetch(texSortPos, j);
    velj = tex1Dfetch(texSortVel, j);
    /*Prepare the seed for the RNG, it must be the same seed
      for pair ij and ji!*/
    if(index<j){
      i0=index;
      j0=j;
    }
    else{
      i0=j;
      j0=index;
    }
    /*Get the random number*/
    randij = randGPU(i0+(ullint)N*j0 +seed, &rng);
    /*Sum the force*/
    force += forceijDPD(pos, posj, veli, velj, randij);
  }
   
  return force;
}


//Kernel to compute the force acting on all particles
__global__ void computeForceDDPD(float4* __restrict__ newForce,
				 uint N, ullint seed){
  /*Travel the particles per sort order*/
  uint ii =  blockIdx.x*blockDim.x + threadIdx.x;
  curandState rng;
  
  //Grid-stride loop
  for(int index = ii; index<N; index += blockDim.x * gridDim.x){
    uint pi = tex1Dfetch(texParticleIndex, index); 
    /*Compute force acting on particle particleIndex[index], index in the new order*/
    float4 pos = tex1Dfetch(texSortPos, index);
    float4 veli= tex1Dfetch(texSortVel, index);
    //float3 veli = vel[pi];
    float4 force = make_float4(0.0f);
    int3 celli = getCell(pos);

    int x,y,z;
    int3 cellj;
    /**Go through all neighbour cells**/
    //For some reason unroll doesnt help here
    for(z=-1; z<=1; z++)
      for(y=-1; y<=1; y++)
	for(x=-1; x<=1; x++){
	  cellj = celli+make_int3(x,y,z);
	  pbc_cell(cellj);	
	  force += forceCellDPD(cellj, index, pos, veli, N, rng, seed);
	}
    /*Write force with the original order*/
    newForce[pi] += force;
   }
}

//CPU kernel caller
void computePairForceDPD(float4 *force,
			 uint N, ullint seed){
  computeForceDDPD<<<GPU_Nblocks, GPU_Nthreads>>>(force,
						  N, seed);
  //cudaCheckErrors("computeForce");
}






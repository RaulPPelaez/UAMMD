/*
Raul P. Pelaez 2016. Interactor GPU kernels and callers.

Functions to compute the force acting on each particle

Neighbour list GPU implementation using hash short with cell index as hash.


References:
http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

TODO:
100- Use Z-order curve as hash instead of cell index to improve memory coherence when traveling the 
     neighbour cells
50- Try bindless textures again.
20- Write coalesced to forces, and use sorted version to integrate, reading sparse from pos and force and writeing coalesced to vel
10- There is no need to reconstruct the neighbour list from scratch each step,
  although computing the force is 50 times as expensive as this right now.

*/

#include<cub/cub/cub.cuh>
#include"InteractorGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"

#include<thrust/device_ptr.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/sort.h>
#include<iostream>


#define BLOCKSIZE 128

using namespace thrust;
using std::cerr;
using std::endl;

__constant__ InteractorParams params; //Simulation parameters in constant memory, super fast access

//Texture references for scattered access
texture<uint> texCellStart, texCellEnd, texParticleIndex;
texture<float4> texSortPos;
texture<float,1 , cudaReadModeElementType> texForce; cudaArray *dF;


uint GPU_Nblocks;
uint GPU_Nthreads;


//Initialize gpu variables 
void initInteractorGPU(InteractorParams m_params, float *potData, size_t potSize,
	     uint *cellStart, uint *cellEnd, uint* particleIndex, uint ncells,
	     float4 *sortPos, uint N){

  /*Precompute some inverses to save time later*/
  m_params.invrc2 = 1.0f/(m_params.rcut*m_params.rcut);
  m_params.invL = 1.0f/m_params.L;
  m_params.invCellSize = 1.0f/m_params.cellSize;

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

  gpuErrchk(cudaMemcpyToArray(dF, 0,0, potData, potSize, cudaMemcpyHostToDevice));

  texForce.normalized = true; //The values are fetched between 0 and 1
  texForce.addressMode[0] = cudaAddressModeClamp; //0 outside [0,1]
  texForce.filterMode = cudaFilterModeLinear; //Linear filtering

  /*Texture binding*/
  gpuErrchk(cudaBindTextureToArray(texForce, dF, channelDesc));


  /*Upload parameters to constant memory*/
  gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(InteractorParams)));

  
  GPU_Nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  GPU_Nblocks  =  N/GPU_Nthreads +  ((N%GPU_Nthreads!=0)?1:0); 
}

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
  return make_int3( (p+0.5f*params.L)*params.invCellSize ); 
}
inline __device__ int3 getCell(float4 p){
  apply_pbc(p); //Reduce to MIC
  return make_int3( (p+0.5f*params.L)*params.invCellSize ); 
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
//Get linear index of a 3D cell
inline __device__ uint getCellIndex(int3 gridPos){
  return gridPos.x+1
    +gridPos.y*params.xcells
    +gridPos.z*params.xcells*params.ycells;
}

//Compute the icell of each particle
__global__ void calcCellIndexD(uint *cellIndex, uint *particleIndex, 
			       const float4 __restrict__ *pos, uint N){
  uint index = blockIdx.x*blockDim.x + threadIdx.x;  
  if(index>N) return;
  float4 p = pos[index];

  int3 gridPos = getCell(p);
  int icell = getCellIndex(gridPos);
 
  cellIndex[index]  = icell;
  //Before ordering by icell the index in the array is the index!
  particleIndex[index] = index;
}

//CPU kernel caller
void calcCellIndex(float4 *pos, uint *cellIndex, uint *particleIndex, uint N){
  calcCellIndexD<<<GPU_Nblocks, GPU_Nthreads>>>(cellIndex, particleIndex, pos, N);
  //  cudaCheckErrors("Calc hash");					   
}

//Sort the particleIndex list by cell index,
// this allows to access the neighbour list of each particle fast and coalesced
void sortCellIndex(uint *&cellIndex, uint *&particleIndex, uint N){
  //This uses the CUB API to perform a radix sort
  //CUB orders by key an array pair array and copies them onto another pair
  //This function stores an internal key/value pair and switches the arrays each time
   static bool init = false;
   static void   *d_temp_storage = NULL;
   static size_t temp_storage_bytes = 0;
   static uint *cellIndex_alt, *particleIndex_alt;

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
     cellIndex= d_keys.Current();
     particleIndex = d_values.Current();

     cellIndex_alt = d_keys.Alternate();
     particleIndex_alt = d_values.Alternate();


   //Thrust is slower and more memory hungry, for it is a higher level call
   // thrust::sort_by_key(device_ptr<uint>(cellIndex),
   // 		      device_ptr<uint>(cellIndex+N),
   // 		      device_ptr<uint>(particleIndex));

   //  cudaCheckErrors("Sort hash");					   
}


//Create CellStart and CellEnd, copy pos onto SortPos
__global__ void reorderAndFindD(float4 *sortPos,
				uint *cellIndex, uint *particleIndex, 
				uint *cellStart, uint *cellEnd,
				float4 *pos,
				uint N){
  uint index = blockIdx.x*blockDim.x + threadIdx.x;
  uint icell, icell2;

  if(index<N){//If my particle is in range  
    icell = cellIndex[index]; //Get my icell
    if(index>0)icell2 = cellIndex[index-1];//Get the previous's part. icell
    else icell2 = 0;
    //If my particle is the first or is in a different cell than the previous
    //my index is the start of a cell
    if(index ==0 || icell != icell2){
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
  //  cudaCheckErrors("Reorder and find");					   
}



//Computes the force between to positions
inline __device__ float4 forceij(const float4 &R1,const float4 &R2){

  float3 r12 = make_float3(R2-R1);

  apply_pbc(r12);

  /*Squared distance between 0 and 1*/
  float r2 = dot(r12,r12)*params.invrc2;
  /*Beyond rcut..*/
  if(r2>=1.0f) return make_float4(0.0f);
  /*Get the force from the texture*/
  float fmod = tex1D(texForce, r2);
  return make_float4(fmod*r12);

  // float invr2 = (1.0f/dot(r12, r12));  
  // if(invr2<params.invrc2) return make_float4(0.0f);
  
  // float invr6 = invr2*invr2*invr2;
  // float invr8 = invr6*invr2;
  
  // float fmod = -48.0f*invr8*invr6+24.0f*invr8;
  // float4 force = make_float4(fmod*r12);
  
  // return force;
 }


//Computes the force acting on particle index from particles in cell cell
__device__ float4 forceCell(const int3 &cell, const uint &index,
			    const float4 &pos){
  uint icell  = getCellIndex(cell);
  /*Index of the first particle in the cell's list*/ 
  uint firstParticle = tex1Dfetch(texCellStart, icell);

  float4 force = make_float4(0.0f);
  /*If there are particles in this cell...*/
  if(firstParticle != 0xffffffff){
    float4 posj;
    /*Index of the last particle in the cell's list*/
    uint lastParticle = tex1Dfetch(texCellEnd, icell);

    /*Because the list is ordered, all the particle indices in the cell are coalescent!*/ 
    for(uint j=firstParticle; j<lastParticle; j++){
      if(j!=index){
	/*Retrieve j pos*/
	posj = tex1Dfetch(texSortPos, j);
	/*Add force */
	force += forceij(pos, posj);
      }
    }
  }
   
  return force;
}


//Kernel to compute the force
__global__ void computeForceD(float4* __restrict__ newForce,
			      const uint* __restrict__ particleIndex, 
			      uint N){
  /*Travel the particles per sort order*/
  uint index =  blockIdx.x*blockDim.x + threadIdx.x;
  if(index>=N) return;
  
  /*Compute force acting on particle particleIndex[index], index in the new order*/
  float4 pos = tex1Dfetch(texSortPos, index);
  
  float4 force = make_float4(0.0f);
  int3 celli = getCell(pos);

  int x,y,z;
  int3 cellj;
  /**Go through all neighbour cells**/
  for(z=-1; z<=1; z++)
    for(y=-1; y<=1; y++)
      for(x=-1; x<=1; x++){
	cellj = celli+make_int3(x,y,z);
	pbc_cell(cellj);	
	force += forceCell(cellj, index, pos);
      }

  /*Write force with the original order*/
  uint pi = tex1Dfetch(texParticleIndex, index); 
  newForce[pi] = force;
}
 
//CPU kernel caller
void computeForce(float4 *sortPos, float4 *force,
		  uint *cellStart, uint *cellEnd,
		  uint *particleIndex, 
		  uint N){
  computeForceD<<<GPU_Nblocks, GPU_Nthreads>>>(force,
					       particleIndex,
					       N);
  //cudaCheckErrors("computeForce");
}


#include<cub/cub.cuh>
#include<curand_kernel.h>
#include"PairForcesAltGPU.cuh"
#include"utils/helper_math.h"
#include"utils/helper_gpu.cuh"

#include<thrust/device_ptr.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/sort.h>
#include<iostream>


typedef unsigned long long int ullint;

#define BLOCKSIZE 128
#define EMPTY_CELL 0xffffffff 

using namespace thrust;
using std::cerr;
using std::endl;

namespace pair_forces_alt_ns{
  __constant__ Params params; //Simulation parameters in constant memory, super fast access
  uint *flagGPU;
  uint flagCPU;
  //Texture references for scattered access
  uint GPU_Nblocks2;
  uint GPU_Nthreads2;
  //texture<float4> texPos;
  cudaTextureObject_t texPos=0, texSortPos=0, texNBL=0;
  cudaStream_t stream;
  //TODO This functions returns an struct with temporal stogare, including temporal arrays for the
  //sorting algorithm etc. So no statics.
  //Initialize gpu variables 
  void initPairForcesAltGPU(Params m_params, float4* pos, float4* sortPos, uint *NBL, uint N, uint maxNPerCell){

    m_params.invL = 1.0f/m_params.L;
    m_params.gridPos2cellIndex = make_int3( 1, m_params.cellDim.x, m_params.cellDim.x*m_params.cellDim.y);
    m_params.getCellFactor = 0.5f*m_params.L*m_params.invCellSize;

    m_params.rmax2 = m_params.rmax*m_params.rmax;
    m_params.rc2 = m_params.rc*m_params.rc;
    m_params.invrc2 = 1.0f/m_params.rc2;
  
    /*Upload parameters to constant memory*/
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));

  
    // gpuErrchk(cudaHostAlloc(&flagCPU, 1, cudaHostAllocPortable));
    // gpuErrchk(cudaHostGetDevicePointer(&flagGPU, flagCPU, 0));
    gpuErrchk(cudaMalloc(&flagGPU, sizeof(uint)));
    flagCPU = 0;
  
    gpuErrchk(cudaMemcpy(flagGPU, &flagCPU, sizeof(uint), cudaMemcpyHostToDevice));


    /*Each particle is asigned a thread*/
    GPU_Nthreads2 = BLOCKSIZE<N?BLOCKSIZE:N;
    GPU_Nblocks2  =  N/GPU_Nthreads2 +  ((N%GPU_Nthreads2!=0)?1:0);



    /*Crearte textures*/

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = pos;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
    resDesc.res.linear.sizeInBytes = N*sizeof(float4);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&texPos, &resDesc, &texDesc, NULL);


    resDesc.res.linear.devPtr = sortPos;
    cudaCreateTextureObject(&texSortPos, &resDesc, &texDesc, NULL);


    resDesc.res.linear.devPtr = NBL;
    resDesc.res.linear.desc = cudaCreateChannelDesc<uint>();
    resDesc.res.linear.sizeInBytes = maxNPerCell*N*sizeof(uint);

    cudaCreateTextureObject(&texNBL, &resDesc, &texDesc, NULL);
  
    cudaStreamCreate(&stream);
  }

  void rebindGPU(uint maxNPerCell, uint N,uint * NBL){
    cudaDestroyTextureObject(texNBL);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = NBL;
    resDesc.res.linear.desc = cudaCreateChannelDesc<uint>();
    resDesc.res.linear.sizeInBytes = maxNPerCell*N*sizeof(uint);

    cudaCreateTextureObject(&texNBL, &resDesc, &texDesc, NULL);


  }
  void updateParamsPairForcesAltGPU(Params m_params){
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }


  /****************************HELPER FUNCTIONS*****************************************/
  //MIC algorithm
  template<typename vecType>
  inline __device__ void apply_pbc(vecType &r){
    r -= floorf(r*params.invL+0.5f)*params.L; //MIC Algorithm
  }

  //Get the 3D cell p is in, just pos in [0,L] divided by ncells(vector) .INT DANGER.
  template<typename vecType>
  inline __device__ int3 getCell(vecType p){
    apply_pbc(p); //Reduce to MIC
    // return  int( (p+0.5L)/cellSize )
    int3 cell = make_int3((p+0.5f*params.L)*params.invCellSize);
    //Anti-Traquinazo guard, you need to explicitly handle the case where a particle
    // is exactly at the box limit, AKA -L/2. This is due to the precision loss when
    // casting int from floats, which gives non-correct results very near the cell borders.
    // This is completly neglegible in all cases, except with the cell 0, that goes to the cell
    // cellDim, wich is catastrophic.
    if(cell.x==params.cellDim.x) cell.x = 0;
    if(cell.y==params.cellDim.y) cell.y = 0;
    if(cell.z==params.cellDim.z) cell.z = 0;
    return cell;
  }

  //Apply pbc to a cell coordinates
  inline __device__ int3 pbc_cell(int3 cell){
    if(cell.x==-1) cell.x = params.cellDim.x-1;
    else if(cell.x==params.cellDim.x) cell.x = 0;

    if(cell.y==-1) cell.y = params.cellDim.y-1;
    else if(cell.y==params.cellDim.y) cell.y = 0;

    if(cell.z==-1) cell.z = params.cellDim.z-1;
    else if(cell.z==params.cellDim.z) cell.z = 0;
  
    return cell;
  }
  //Get linear index of a 3D cell, from 0 to ncells-1
  inline __device__ uint getCellIndex(int3 gridPos){
    return dot(gridPos, params.gridPos2cellIndex);
    // return gridPos.x
    //   +gridPos.y*params.xcells
    //   +gridPos.z*params.xcells*params.ycells;
  }



  // __global__ void calcCellIndexD(uint *cellIndex, uint *particleIndex, float4 *pos, uint N){
  //   uint i = blockIdx.x*blockDim.x + threadIdx.x;
  //   if(i>=N) return;
  //   float4 p = pos[i];
  //   cellIndex[i] = getCellIndex(getCell(p));
  //   particleIndex[i] = i;
  // }
  void sortCellIndex2(uint *&cellIndex, uint *&particleIndex, uint N){
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
    //   gpuErrchk(cudaBindTexture(NULL, texParticleIndex,   particleIndex,   (N+1)*sizeof(uint)));
  }

  // __global__ void findCellStartEnd(uint *cellIndex, uint *particleIndex, uint N,
  // 				 uint *cellSize, uint *cellEnd){
  //   uint index = blockIdx.x*blockDim.x + threadIdx.x; //particle
  //   uint icell, icell2;
  //   if(index<N){//If my particle is in range
  //     icell = cellIndex[index]; //Get my icell
  //     //TODO VVV this is a target for shared memory
  //     if(index>0)icell2 = cellIndex[index-1];//Get the previous part.'s icell
  //     else icell2 = 0;
  //     //If my particle is the first or is in a different cell than the previous
  //     //my index is the start of a cell
  //     if(index ==0 || icell != icell2){
  //       //Then my particle is the first of my cell
  //       cellSize[icell] = index;
  //       //If my index is the start of a cell, it is also the end of the previous
  //       //Except if I am the first one
  //       if(index>0)
  // 	cellEnd[icell2] = index;
  //     }
  //     //If I am the last particle my cell ends 
  //     if(index == N-1) cellEnd[icell] = index+1;
  //   }



  // }
  // __global__ void fillCELL(uint *CELL,
  // 			 uint *cellIndex, uint *particleIndex,
  // 			 uint *cellSize, uint *cellEnd, uint N, uint maxNPerCell){
  //   uint cell = blockIdx.x;

  //   uint firstParticle = cellSize[cell];
  //   uint lastParticle = cellEnd[cell];

  //   if(threadIdx.x==0) cellSize[cell] = lastParticle-firstParticle;
  //   if(firstParticle==EMPTY_CELL) return;

  //   for(uint i = threadIdx.x; i<maxNPerCell; i+=blockDim.x){
  //     //TODO VVV target for shared memory
  //     if((firstParticle+i)>=lastParticle) return;  //TODO maybe this should go before icell
  //     //TODO VVV target for texture memory
  //     CELL[cell*maxNPerCell+i] = particleIndex[firstParticle+i]; 
  //   }
  // }


  __global__ void fillCELLAtomic(uint *CELL, uint *cellSize, float4 *pos, uint N, uint ncells, uint maxNPerCell, uint *flag){

    uint i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=N) return;
  
    float4 p = pos[i];

    uint icell = getCellIndex(getCell(p));

    uint size = atomicInc(&cellSize[icell], 0xffFFffFF);
  
    if(size>=maxNPerCell){
      atomicMax(flag, 1U); //*flag=1; also works
      return;
    }

    CELL[icell*maxNPerCell+size] = i;
  }

  inline __device__ uint encodeMorton(uint i){
  
    uint x = i;
  
    x &= 0x3ff;
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x300f00f;
    x = (x | x << 4) & 0x30c30c3;
    x = (x | x << 2) & 0x9249249;
    return x;
  }

  inline __device__ uint mortonHash(const int3 &cell){
    return encodeMorton(cell.x) | (encodeMorton(cell.y) << 1) | (encodeMorton(cell.z) << 2);
  }


  __global__ void calcHashIndexD( float4 *pos, uint *hashIndex, uint *particleIndex, uint N){

    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;
  
    float4 p = pos[i];
  
    int3 cell = getCell(p);

    uint hash = mortonHash(cell);
  
    hashIndex[i] = hash;
    particleIndex[i] = i;
  }

  __global__ void reorderPosD(cudaTextureObject_t pos, float4* sortPos, uint *particleIndex, uint N){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;
  
    sortPos[i] = tex1Dfetch<float4>(pos, particleIndex[i]);
  }

  void orderByHash(float4* pos, float4* sortPos, uint *&hashIndex, uint *&particleIndex, uint N){
  
    calcHashIndexD<<<GPU_Nblocks2, GPU_Nthreads2>>>(pos, hashIndex, particleIndex, N);
  
    sortCellIndex2(hashIndex, particleIndex, N);

    reorderPosD<<<GPU_Nblocks2, GPU_Nthreads2>>>(texPos, sortPos, particleIndex, N);

  }

  __global__ void makeNeighbourListD(uint *CELL, uint *NBL, uint *Nneigh,
				     cudaTextureObject_t texPos2, uint N, uint ncells, uint maxNPerCell,
				     uint *flag){
    //All dynamic shared memory must go in one array
    extern __shared__ uint sm[];
    //extern __shared__ float4 rj[];
    uint *K    = sm;
    float4 *rj = (float4*)&sm[blockDim.x];
  
    uint icell = blockIdx.x;
  
    uint tid = threadIdx.x; 
    uint i = CELL[icell*maxNPerCell+tid];
    uint nneigh = 0;
  
    //TODO VVV target for texture memory
    float4 ri = tex1Dfetch<float4>(texPos2, i);//make_float4(0);

  
    //TODO VVV there must be a better way to go through the 27 neighbour cells
    int3 cell = {icell%params.cellDim.x,
		 (icell/params.cellDim.x)%params.cellDim.y,
		 icell/(params.cellDim.x*params.cellDim.y)};
  
    uint neigh_icell;
    for(int x=-1; x<=1; x++)
      for(int y=-1; y<=1; y++)
	for(int z=-1; z<=1; z++){
	  int3 vcell = pbc_cell(cell + make_int3(x,y,z));
	  neigh_icell = getCellIndex(vcell);
	  __syncthreads();
	  K[tid] = CELL[neigh_icell*maxNPerCell+tid];
	  rj[tid] = tex1Dfetch<float4>(texPos2, K[tid]);//make_float4(0);
	  __syncthreads();
	  if(i!=EMPTY_CELL){
	    for(uint j=0; j<maxNPerCell; j++){
	      if(K[j]==EMPTY_CELL) break;
	      float3 rij = make_float3(rj[j]-ri);
	      apply_pbc(rij);
	      if(dot(rij, rij)<params.rmax2 && K[j] != i){
		if(nneigh>=maxNPerCell){
		  atomicMax(flag, 1U); //*flag=1; also works
		}
		else{
		  NBL[nneigh*N+i] = K[j];
		  nneigh++;
		}
	      }
	    }
	  }
	}
    if(i!=EMPTY_CELL) Nneigh[i] = nneigh;
  }

  bool makeNeighbourListGPU2(uint *&cellIndex, uint *cellSize, uint *&particleIndex,
			     float rmax,
			     uint *CELL, uint ncells,
			     uint *NBL, uint *Nneigh,
			     float4 *pos, float4* old_pos, float4 *sortPos,
			     uint N, uint maxNPerCell){

    cudaMemsetAsync(CELL, EMPTY_CELL, ncells*maxNPerCell*sizeof(uint), stream);
    cudaMemsetAsync(cellSize, 0, ncells*sizeof(uint));
    cudaMemcpyAsync(old_pos, pos, N*sizeof(float4), cudaMemcpyDeviceToDevice, stream);
  
    // calcCellIndexD<<<GPU_Nblocks2, GPU_Nthreads2>>>(cellIndex, particleIndex, pos, N);
    // sortCellIndex2(cellIndex, particleIndex, N);
    // findCellStartEnd<<<GPU_Nblocks2, GPU_Nthreads2>>>(cellIndex, particleIndex, N,
    // 						    cellSize, cellEnd);
    // fillCELL<<<ncells, maxNPerCell>>>(CELL, cellIndex, particleIndex,
    // 				 cellSize, cellEnd, N,  maxNPerCell);
  
    orderByHash(pos, sortPos, cellIndex, particleIndex, N);
  
    cudaStreamSynchronize(stream);
    fillCELLAtomic<<<GPU_Nblocks2, GPU_Nthreads2>>>(CELL, cellSize, sortPos, N, ncells, maxNPerCell, flagGPU);
  
    cudaMemcpy(&flagCPU, flagGPU, sizeof(uint), cudaMemcpyDeviceToHost);

    if(flagCPU){
      flagCPU = 0;
      cudaMemcpy(flagGPU, &flagCPU, sizeof(uint), cudaMemcpyHostToDevice);    
      return false;
    }

    makeNeighbourListD<<<ncells,
      maxNPerCell,
      maxNPerCell*(sizeof(float4)+sizeof(uint))>>>(CELL, NBL, Nneigh,
						   texSortPos,
						   N, ncells, maxNPerCell, flagGPU);

  
    cudaMemcpy(&flagCPU, flagGPU, sizeof(uint), cudaMemcpyDeviceToHost);

    if(flagCPU){
      flagCPU = 0;
      cudaMemcpy(flagGPU, &flagCPU, sizeof(uint), cudaMemcpyHostToDevice);    
      return false;
    }


    return true;
  }


  inline __device__ float4 forceij(const float4 &ri, const float4 &rj){
  
    float3 r12 = make_float3(rj-ri);
    apply_pbc(r12);

    /*Squared distance between 0 and 1*/
    float r2 = dot(r12,r12);
    float r2c = r2*params.invrc2;
    /*Beyond rcut..*/
    /*Check if i==j. This way reduces warp divergence and its faster than checking i==j outside*/
    if(r2c==0.0f || r2c > 1.0f) return make_float4(0.0f);  //Both cases handled in texForce
  
    /*Get the force from the texture*/
    //float fmod = tex1D(texForce, r2c);
    float invr2 = 1.0f/r2;
    float invr6 = invr2*invr2*invr2;
    float invr8 = invr6*invr2;
    //float E =  2.0f*invr6*(invr6-1.0f);
    float fmod = -48.0f*invr8*invr6+24.0f*invr8;
  
    return make_float4(fmod*r12);
  }
  __global__ void computePairForceD(__restrict__ const uint *NBL,
				    __restrict__ const uint *Nneigh,
				    __restrict__ const uint *particleIndex,
				    __restrict__ float4 *force,
				    const uint N, const uint maxNPerCell,
				    const cudaTextureObject_t texPos2){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;
    uint unsort_index = particleIndex[i];
    float4 f = make_float4(0.0f);
    float4 ri = tex1Dfetch<float4>(texPos2, i);// pos[i]; //Coalesced //

  
    uint nneigh = Nneigh[i]; //Coalesced
    for(uint j = 0; j<nneigh; j++){
      uint k = NBL[j*N+i]; //Coalesced
      float4 rj =  tex1Dfetch<float4>(texPos2, k);//pos[k];
      f += forceij(ri, rj);
    }

    force[unsort_index] += f;
  }

  __device__ float warp_reduce(uint NT, uint tid, float x){


    for (int dest_count = NT/2; dest_count >= 1; dest_count /= 2){

      x += __shfl_down(x, dest_count, NT);

    }
    return x;

  }
  __global__ void computePairForceDTPP(__restrict__ const uint *NBL,
				       __restrict__ const uint *Nneigh,
				       __restrict__ const uint *particleIndex,
				       __restrict__ float4 *force,
				       const uint N, const uint maxNPerCell,
				       const cudaTextureObject_t texPos2,
				       const cudaTextureObject_t texNBL2,
				       uint tpp){
    uint i = blockIdx.x*(blockDim.x/tpp) + threadIdx.x/tpp;
    if(i>=N) return;
  

    float4 f = make_float4(0.0f);
    float4 ri = tex1Dfetch<float4>(texPos2, i);// pos[i]; //Coalesced //

  
    uint nneigh = Nneigh[i]; //Coalesced
    uint j = threadIdx.x%tpp;
    while(j<nneigh){
      uint k = tex1Dfetch<uint>(texNBL2, j*N+i); //NBL[j*N+i];//
      j+=tpp;
      float4 rj =  tex1Dfetch<float4>(texPos2, k);//pos[k];
      f += forceij(ri, rj);
    }

    f.x = warp_reduce(tpp, threadIdx.x%tpp, f.x);
    f.y = warp_reduce(tpp, threadIdx.x%tpp, f.y);
    f.z = warp_reduce(tpp, threadIdx.x%tpp, f.z);

  
    if(threadIdx.x%tpp == 0){
      uint unsort_index = particleIndex[i];
      force[unsort_index] += f;
    }
  
  }
 
  void computePairForce(uint *NBL, uint *Nneigh, uint *particleIndex,
			float4 *pos, float4 *sortPos, float4* force, uint N, uint maxNPerCell){

    reorderPosD<<<GPU_Nblocks2, GPU_Nthreads2>>>(texPos, sortPos, particleIndex, N);
    
    computePairForceD<<<GPU_Nblocks2, GPU_Nthreads2>>>(NBL, Nneigh, particleIndex, force, N, maxNPerCell, texSortPos);

  }


  __global__ void checkBinningCellsD(float4 *pos, float4* old_pos, uint *flag, uint N, float rthres2){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;

    float4 p = pos[i];
    float4 op = old_pos[i];
    float3 r12 = make_float3(p-op);

    if(dot(r12, r12)>rthres2)
      atomicMax(flag, 1U);

  }

  bool checkBinningCells(float4 *old_pos,  float4 *cur_pos, uint N, float rthreshold){
    checkBinningCellsD<<<GPU_Nblocks2, GPU_Nthreads2>>>(cur_pos, old_pos, flagGPU, N, rthreshold*rthreshold);
    cudaMemcpy(&flagCPU, flagGPU, sizeof(uint), cudaMemcpyDeviceToHost);
    if(flagCPU){
      flagCPU = 0;
      cudaMemcpy(flagGPU, &flagCPU, sizeof(uint), cudaMemcpyHostToDevice);
      return true;
    }
    return false;

  }
}

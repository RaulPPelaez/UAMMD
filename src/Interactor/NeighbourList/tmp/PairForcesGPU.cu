/*Raul P. Pelaez 2016. Short range pair forces Interactor GPU callers and kernels.

  The Neighbour list is constructed in the GPU as follows:
  
  1-Compute a hash for each particle based on its cell. Store in particleHash, also fill particleIndex with the index of each particle (particleIndex[i] = i)
  2-Sort particleIndex based on particleHash (sort by key). This way the particles in a same cell are one after the other in particleIndex. The Morton hash also improves the memory acces patter in the GPU.
  3-Fill cellStart and cellEnd with the indices of particleIndex in which a cell starts and ends. This allows to identify where all the [indices of] particles in a cell are in particleIndex, again, one after the other.
  
  The transversal of this cell list is done by transversing, for each particle, the 27 neighbour cells of that particle's cell. You can make use of this by creating a transversable, see forceTransversable
  

  Force is evaluated using table lookups (with texture memory)

  Throughout this file, all __device__ and __global__ functions have access to the params and, in the case of DPD, paramsDPD structs in constant memory. See PairForcesGPU.cuh for a list of the available information in this structures.

  You can add new information to constant params by declaring it in PairForcesGPU.cuh and initializing it either in the initGPU function  before uploading or in the PairForces constructor, before calling initGPU.

  References:
  http://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

  TODO:
  100- Implement many threads per particle in force compute
  100- Make number of blocks and threads to autotune
  100- Improve the transversing of the 27 neighbour cells
  90- Implement energy and virial compute in PairForcesDPD, maybe take it to another file
  80- General functions like apply_pbc should be made global to ease development.
  10- Find a way to properly handle the alternate arrays in sortCellIntex
*/

#include<cub/cub.cuh>
#include<curand_kernel.h>
#include"PairForcesGPU.cuh"
#include"globals/defines.h"
#include"globals/globals.h"
#include<thrust/device_ptr.h>
#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/sort.h>
#include<iostream>
#include"utils/GPUutils.cuh"


extern SystemInfo sysInfo;

#define BLOCKSIZE 128

using namespace thrust;
using std::cerr;
using std::endl;

namespace pair_forces_ns{
  __constant__ Params params; //Simulation parameters in constant memory, super fast access
  __constant__ ParamsDPD paramsDPD; //Simulation parameters in constant memory, super fast access
  __constant__ ullint seedGPU; //seed of the rng for DPD


  texture<float,1, cudaReadModeElementType> texForce, texEnergy;
  cudaArray *dF, *dE;
  
  uint GPU_Nblocks;
  uint GPU_Nthreads;
  
  
  //Initialize gpu variables 
  void initGPU(Params &m_params, Params *&d_params, uint N, size_t potSize){
    /*Precompute some inverses to save time later*/
    m_params.invrc2_pot = 1.0/(m_params.rcut_pot*m_params.rcut_pot);
    m_params.invrc_pot = 1.0/(m_params.rcut_pot); 

    m_params.invCellSize = 1.0/m_params.cellSize;
    m_params.gridPos2CellIndex = make_int3( 1,
					    m_params.cellDim.x,
					    m_params.cellDim.x*m_params.cellDim.y);    

    if(gcnf.D2){
      m_params.invCellSize.z = 0.0;
    }
    /*Only new architectures support texture objects*/
    if(sysInfo.cuda_arch>210){
      if(!m_params.texPos.tex || !m_params.texSortPos.tex || !m_params.texCellStart.tex ||
	 !m_params.texCellEnd.tex || !m_params.texForce.tex || !m_params.texEnergy.tex){
	cerr<<"Problem setting up textures!!"<<endl;
	exit(1);
      }
    }
    else{ /*In the old arch 20 the code uses textures only for force and energy*/     
      /*Create and bind force texture, this needs interpolation*/
      cudaChannelFormatDesc channelDesc;
      channelDesc = cudaCreateChannelDesc(32, 0,0,0, cudaChannelFormatKindFloat);

      gpuErrchk(cudaMallocArray(&dF,
				&channelDesc,
				potSize/sizeof(float),1));
      gpuErrchk(cudaMallocArray(&dE,
				&channelDesc,
				potSize/sizeof(float),1));

      gpuErrchk(cudaMemcpyToArray(dF, 0,0,
				  (float*)m_params.texForce.d_ptr, potSize, cudaMemcpyDeviceToDevice));
      gpuErrchk(cudaMemcpyToArray(dE, 0,0,
				  (float*)m_params.texEnergy.d_ptr, potSize, cudaMemcpyDeviceToDevice));

      texForce.normalized = true; //The values are fetched between 0 and 1
      texForce.addressMode[0] = cudaAddressModeClamp; //0 outside [0,1]
      texForce.filterMode = cudaFilterModeLinear; //Linear filtering
      texEnergy.normalized = true; //The values are fetched between 0 and 1
      texEnergy.addressMode[0] = cudaAddressModeClamp; //0 outside [0,1]
      texEnergy.filterMode = cudaFilterModeLinear; //Linear filtering

      /*Texture binding*/
      gpuErrchk(cudaBindTextureToArray(texForce, dF, channelDesc));
      gpuErrchk(cudaBindTextureToArray(texEnergy, dE, channelDesc));
    }


    /*Upload params to global memory*/
    gpuErrchk(cudaMalloc(&d_params, sizeof(Params)));
    gpuErrchk(cudaMemcpy(d_params, &m_params, sizeof(Params), cudaMemcpyHostToDevice));
    
    /*Upload parameters to constant memory*/
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));

    /*Each particle is asigned a thread*/
    GPU_Nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
    GPU_Nblocks  =  N/GPU_Nthreads +  ((N%GPU_Nthreads!=0)?1:0); 
  }


  void initDPDGPU(ParamsDPD &m_params){
    gpuErrchk(cudaMemcpyToSymbol(paramsDPD, &m_params, sizeof(ParamsDPD)));
  }

  /*TODO you have to update textures too in arch<350*/
  /*Reupload the parameters to constant memory from CPU*/
  void updateParams(Params m_params){
    gpuErrchk(cudaMemcpyToSymbol(params, &m_params, sizeof(Params)));
  }
  /*Reupload the parameters to constant memory from GPU*/
  void updateParamsFromGPU(Params *d_params){
    gpuErrchk(cudaMemcpyToSymbol(params, d_params, sizeof(Params), 0, cudaMemcpyDeviceToDevice));
  }

  
  /****************************HELPER FUNCTIONS*****************************************/  
  //Get the 3D cell p is in, just pos in [0,L] divided by ncells(vector) .INT DANGER.

  inline __device__ int3 getCell(real3 r){    
    apply_pbc(r); //Reduce to MIC
    // return  int( (p+0.5L)/cellSize )
    int3 cell = make_int3((r+real(0.5)*gcnfGPU.L)*params.invCellSize);
    //Anti-Traquinazo guard, you need to explicitly handle the case where a particle
    // is exactly at the box limit, AKA -L/2. This is due to the precision loss when
    // casting int from floats, which gives non-correct results very near the cell borders.
    // This is completly neglegible in all cases, except with the cell 0, that goes to the cell
    // cellDim, which is catastrophic.
    //Doing the previous operation in double precision (by changing 0.5f to 0.5) also works, but it is a bit of a hack and the performance appears to be the same as this.
    //TODO: Maybe this can be skipped if the code is in double precision mode
    if(cell.x==params.cellDim.x) cell.x = 0;
    if(cell.y==params.cellDim.y) cell.y = 0;
    if(cell.z==params.cellDim.z) cell.z = 0;
    return cell;
  }

  //Apply pbc to a cell coordinates
  inline __device__ void pbc_cell(int3 &cell){
    if(cell.x==-1) cell.x = params.cellDim.x-1;
    else if(cell.x==params.cellDim.x) cell.x = 0;

    if(cell.y==-1) cell.y = params.cellDim.y-1;
    else if(cell.y==params.cellDim.y) cell.y = 0;

    if(cell.z==-1) cell.z = params.cellDim.z-1;
    else if(cell.z==params.cellDim.z) cell.z = 0;
  }
  
  //Get linear index of a 3D cell, from 0 to ncells-1
  inline __device__ uint getCellIndex(int3 gridPos){
    return dot(gridPos, params.gridPos2CellIndex);
  }


  /*Interleave a 10 bit number in 32 bits, fill one bit and leave the other 2 as zeros.*/
  inline __device__ uint encodeMorton(uint i){
    uint x = i;
    x &= 0x3ff;
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x300f00f;
    x = (x | x << 4) & 0x30c30c3;
    x = (x | x << 2) & 0x9249249;
    return x;
  }
  /*Fuse three 10 bit numbers in 32 bits, producing a Z order Morton hash*/
  inline __device__ uint mortonHash(const int3 &cell){
    return encodeMorton(cell.x) | (encodeMorton(cell.y) << 1) | (encodeMorton(cell.z) << 2);
  }
  
  /****************************************************************************************/


  /*Assign a hash to each particle from its cell index*/
  __global__ void calcHashD(uint __restrict__ *particleHash, uint __restrict__ *particleIndex, 
			    const real4 __restrict__ *pos, uint N){
    uint index = blockIdx.x*blockDim.x + threadIdx.x;  
    if(index>=N) return;
    real3 p = make_real3(pos[index]);

    int3 cell = getCell(p);
    /*The particleIndex array will be sorted by the hashes, any order will work*/
    uint hash = mortonHash(cell);//getCellIndex(cell);
    /*Before ordering by hash the index in the array is the index itself*/
    particleIndex[index] = index;
    particleHash[index]  = hash;
  }  
  //CPU kernel caller
  void calcHash(real4 *pos, uint *particleHash, uint *particleIndex, uint N){
    calcHashD<<<GPU_Nblocks, GPU_Nthreads>>>(particleHash, particleIndex, pos, N);
    //cudaCheckErrors("Calc hash");					   
  }

  /*Sort the particleIndex list by hash*/
  // this allows to access the neighbour list of each particle in a more coalesced manner
  //Each time this is called, the pointers particleHash and particleIndex are swapped with local ones
  void sortCellHash(uint *&particleHash, uint *&particleIndex, uint N){
    //This uses the CUB API to perform a radix sort
    //CUB orders by key an array pair and copies them onto another pair
    //This function stores an internal key/value pair and switches the arrays each time its called
    static bool init = false;
    static void *d_temp_storage = NULL;
    static size_t temp_storage_bytes = 0; //Additional storage needed by cub
    static uint *particleHash_alt = NULL, *particleIndex_alt = NULL; //Additional key/value pair

    /**Initialize CUB at first call**/
    if(!init){
      /*Allocate temporal value/key pair*/
      gpuErrchk(cudaMalloc(&particleHash_alt,  N*sizeof(uint)));
      gpuErrchk(cudaMalloc(&particleIndex_alt, N*sizeof(uint)));    
      /*On first call, this function only computes the size of the required temporal storage*/
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				      particleHash, particleHash_alt,
				      particleIndex, particleIndex_alt,
				      N);
      /*Allocate temporary storage*/
      gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      init = true;
    }

    /**Perform the Radix sort on the index/hash pair**/
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
				      particleHash, particleHash_alt,
				      particleIndex, particleIndex_alt,
				      N);
    /**Swap the references**/
    swap(particleHash, particleHash_alt);
    swap(particleIndex, particleIndex_alt);

    // thrust::stable_sort_by_key(device_ptr<uint>(particleHash),
    // 			device_ptr<uint>(particleHash+N),
    // 			device_ptr<uint>(particleIndex));
    //cudaCheckErrors("Sort hash");					   
  }

  /*This kernel fills sortPos with the positions in pos, acording to the indices in particleIndex*/
  __global__ void reorderPosD(real4 __restrict__ *sortPos, const real4 __restrict__ *pos,
			      const uint* __restrict__ particleIndex, uint N){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;

    uint sort_index = particleIndex[i]; //Coalesced

    sortPos[i] = tex1Dfetch<real4>(params.texPos, sort_index);
  }
  /*Same as above, but reordering vel aswell*/
  __global__ void reorderPosVelD(real4 *sortPos,
				 real4 *pos,
				 real4* sortVel,
				 real3 * vel,
				 uint* particleIndex, uint N){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;

    uint sort_index = particleIndex[i]; //Coalesced


    sortPos[i] = tex1Dfetch<real4>(params.texPos, sort_index);
    /*TODO: Texture this aswell*/
    sortVel[i] = make_real4(vel[sort_index]);
  }
  
  /*Fill CellStart and CellEnd*/
  __global__ void fillCellListD(const real4 __restrict__ *sortPos,
				uint *cellStart, uint *cellEnd,
				uint N){
    /*A thread per particle*/
    uint i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i<N){//If my particle is in range
      uint icell, icell2;
      /*Get my icell*/
      icell = getCellIndex(getCell(make_real3(sortPos[i])));
      
      /*Get the previous part.'s icell*/
      if(i>0){ /*Shared memory target VVV*/
	icell2 = getCellIndex(getCell(make_real3(sortPos[i-1])));
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

  /*Create the Cell List from scratch in the GPU*/
  void makeCellList(real4 *pos, real4 *sortPos, //real4 *oldPos,
		    uint *&particleIndex, uint *&particleHash,
		    uint *cellStart, uint *cellEnd,
		    uint N, uint ncells){
    cudaMemset(cellStart, 0xffffffff, ncells*sizeof(uint));

    //cudaMemcpy(oldPos, pos, N*sizeof(real4), cudaMemcpyDeviceToDevice);
    
    calcHashD<<<GPU_Nblocks, GPU_Nthreads>>>(particleHash, particleIndex, pos, N);    
    sortCellHash(particleHash, particleIndex, N);
    
    reorderPosD<<<GPU_Nblocks, GPU_Nthreads>>>(sortPos, pos, particleIndex, N);
    /*This fills cellStart and cellEnd*/
    fillCellListD<<<GPU_Nblocks, GPU_Nthreads>>>(sortPos, cellStart, cellEnd, N);
    
  }
  
  void makeCellListDPD(real4 *pos, real3* vel,  real4 *sortPos, real4 *sortVel,
		       uint *&particleIndex, uint *&particleHash,
		       uint *cellStart, uint *cellEnd,
		       uint N, uint ncells){
    
    cudaMemset(cellStart, 0xffffffff, ncells*sizeof(uint));

    calcHashD<<<GPU_Nblocks, GPU_Nthreads>>>(particleHash, particleIndex, pos, N);
    
    sortCellHash(particleHash, particleIndex, N);

    reorderPosVelD<<<GPU_Nblocks, GPU_Nthreads>>>(sortPos, pos, sortVel, vel, particleIndex, N);

    fillCellListD<<<GPU_Nblocks, GPU_Nthreads>>>(sortPos, cellStart, cellEnd, N);
    
  }

  //TODO The naming and explanation of this function
  /*Transverses all the neighbour particles of each particle using the cell list and computes a quantity as implemented by Transverser. Each thread goes through all the neighbours of a certain particle(s)(index), transversing its 27 neighbour cells*/
  /*Computes a quantity determined by Transverser, which is a class that must implement the following methods:
    T zero() -> returns the initial value of the quantity, in whatever type
    T compute(ParticleInfo r1, ParticleInfo r2) -> compute the quantity depending of the pair positions/types/etc
    void set(uint index, T quantity) -> sum the total quantity on particle index to global memory

    This quantity "T" can be anything that can be sum. i.e a real4 and compute the force
                       or a real and compute the energy or a general struct containing any info (which must have an operator += overloaded)...
    ParticleInfo can be a simple type like real4 containing the position or a general struct defined in the Transverser class containing anything (pos and vel i.e)
    */  
  template<class Transverser>
  __global__ void transverseListD(Transverser T, 
				  const uint* __restrict__ particleIndex,
				  const uint* __restrict__ cellStart,
				  const uint* __restrict__ cellEnd,
				  uint N){
    uint ii =  blockIdx.x*blockDim.x + threadIdx.x;

    //Grid-stride loop
    for(int index = ii; index<N; index += blockDim.x * gridDim.x){
      /*Compute force acting on particle particleIndex[index], index in the new order*/
      /*Get my particle's.data*/
      auto pinfoi = T.getInfo(index);

      /*Initial value of the quantity,
	mostly just a hack so the compiler can guess the type of the quantity*/
      auto quantity = T.zero();
      
      int3 celli = getCell(make_real3(pinfoi.pos));

      int x,y,z;
      int3 cellj;
      /**Go through all neighbour cells**/
      //For some reason unroll doesnt help here
      int zi = -1; //For the 2D case
      int zf = 1;
      if(gcnfGPU.D2){
	zi = zf = 0;
      }
      for(z=zi; z<=zf; z++)
	for(y=-1; y<=1; y++)
	  for(x=-1; x<=1; x++){
	    cellj = celli + make_int3(x,y,z);
	    pbc_cell(cellj);

	    uint icell  = getCellIndex(cellj);
	    /*Index of the first particle in the cell's list*/
	    uint firstParticle = tex1Dfetch<uint>(params.texCellStart, icell);
	    if(firstParticle ==0xffFFffFF) continue; /*Continue only if there are particles in this cell*/
	    /*Index of the last particle in the cell's list*/
	    uint lastParticle = tex1Dfetch<uint>(params.texCellEnd, icell);
	    uint nincell = lastParticle-firstParticle;
	    /*Because the list is ordered, all the particle indices in the cell are coalescent!*/
	    for(uint j=0; j<nincell; j++){
	      /*Retrieve j info*/
	      auto pinfoj = T.getInfo(j+firstParticle);
	      /*Add quantity*/
	      quantity += T.compute(pinfoi, pinfoj);	      

	    }
	    
	  }
      /*Write quantity with the original order to global memory*/
#if __CUDA_ARCH__>=350
      uint pi = __ldg(particleIndex+index);
#else
      uint pi = particleIndex[index]; 
#endif
      T.set(pi, quantity);
    }
    
  }

  //NO FUNCIONA, CUB DOES SOMETHING ODD
#define tpp 1
  template<class Transverser>
  __global__ void transverseListDtpp(Transverser T, 
				     const uint* __restrict__ particleIndex,
				     const uint* __restrict__ cellStart,
				     const uint* __restrict__ cellEnd,
				     uint N){
    uint index =  blockIdx.x*(blockDim.x/tpp) + threadIdx.x/tpp;
    bool active = true;

    if(index>=N) active = false;

    auto quantity = T.zero();

    typedef cub::WarpReduce<decltype(quantity), tpp> WarpReduce;
    constexpr int particles_per_block = BLOCKSIZE/tpp;
    __shared__ typename WarpReduce::TempStorage temp_storage[particles_per_block];

    
    if(active){
      auto pinfoi = T.getInfo(index);
      int3 celli = getCell(make_real3(pinfoi.pos));
   
      int x,y,z;
      int3 cellj;
      /**Go through all neighbour cells**/
      //For some reason unroll doesnt help here
      int zi = -1; //For the 2D case
      int zf = 1;
      if(gcnfGPU.D2){
	zi = zf = 0;
      }
      for(z=zi; z<=zf; z++)
	for(y=-1; y<=1; y++)
	  for(x=-1; x<=1; x++){
	    cellj = celli + make_int3(x,y,z);
	    pbc_cell(cellj);

	    uint icell  = getCellIndex(cellj);
	    /*Index of the first particle in the cell's list*/
	    uint firstParticle = tex1Dfetch<uint>(params.texCellStart, icell);
	    if(firstParticle != 0xffFFffFF) continue; /*Continue only if there are particles in this cell*/
	      /*Index of the last particle in the cell's list*/
	      uint lastParticle = tex1Dfetch<uint>(params.texCellEnd, icell);
	      uint nincell = lastParticle-firstParticle;
	      /*Because the list is ordered, all the particle indices in the cell are coalescent!*/
	      for(uint j=threadIdx.x%tpp; j<nincell+tpp-nincell%tpp; j+=tpp){
		if(j<nincell){
		  /*Retrieve j info*/
		  auto pinfoj = T.getInfo(j+firstParticle);
		  /*Add quantity*/
		  quantity += T.compute(pinfoi, pinfoj);
		}
	      }
	    }
    }

    int warp_id = threadIdx.x/particles_per_block;
    auto total_quantity = WarpReduce(temp_storage[warp_id]).Sum(quantity);
    
    if(active && threadIdx.x%tpp==0){
      /*Write quantity with the original order to global memory*/
#if __CUDA_ARCH__>=350
      uint pi = __ldg(particleIndex+index);
#else
      uint pi = particleIndex[index]; 
#endif
      T.set(pi, total_quantity);
    }
    
  }
  
  
  /***************************************FORCE*****************************/
  
  //tags: force compute force function forceij
  /*A helper class that holds the address of the force array (in device) and
    computes the force between two particles*/
  /*It also updates the global array newForce with the total force acting on particle pi*/
  /*This helper class can be passed as an argument to transverseListD, which will apply the compute
    function to every neighbouring particle pair in the system*/
  /*In order to compute any other quantity create a class like this, implementing the same functions
    but with any desired type, instead of real4 as in this case*/
  template<bool many_types>
  class forceTransverser{
  public:
    struct ParticleInfo{
      real4 pos;
    };
    /*I need the device pointer to force*/
    forceTransverser(real4 *newForce):newForce(newForce){};
    /*Compute the force between two positions*/
    inline __device__ real4 compute(const ParticleInfo &R1,const ParticleInfo &R2){
      real3 r12 = make_real3(R2.pos)-make_real3(R1.pos);
      apply_pbc(r12);
      /*Squared distance*/
      /*Squared distance between 0 and 1*/
      const real r2 = dot(r12,r12);
      real r2c = r2*(float)params.invrc2_pot;

      /*Reescale for current type pair*/
      real epsilon = real(1.0); 
      if(many_types){
       	const int ti= (int)(R1.pos.w+0.5f);
       	const int tj= (int)(R2.pos.w+0.5f);
       	const real2 pot_params = params.potParams[ti+tj*params.ntypes];
       	const real sigma2 = pot_params.x*pot_params.x;
       	epsilon = pot_params.y;
	/*STORE invsig2 instead, and directly rescale r2c, this really slows down rt*/
       	r2c /= sigma2;
       }
      /*Get the force from the texture*/
#if __CUDA_ARCH__>210
       const real3 f = (epsilon*(real) tex1D<float>(params.texForce.tex, r2c))*r12;
#else /*Use texture reference*/
       const real3 f = (epsilon*(real) tex1D(texForce, r2c))*r12;
#endif
      
      // /*Both cases handled in texForce*/
      // /*Check if ai==j. This way reduces warp divergence and its faster than checking i==j outside*/
      // if(r2c==real(0.0)) return make_real4(real(0.0));  
      // /*Beyond rcut..*/
      // else if(r2c>=real(1.0)) return make_real4(real(0.0));

      // real3 f;
      // if(r2<params.rcut_pot*params.rcut_pot){
      // 	real invr2 = real(1.0)/r2;
      // 	real invr6 = invr2*invr2*invr2;
      // 	f = -epsilon*real(24.0)*invr2*invr6*(real(2.0)*invr6-real(1.0))*r12;       
      // }
      // else f=make_real3(real(0.0));
      
      return  make_real4(f, real(0.0));
    }
    /*Update the force acting on particle pi, pi is in the normal order*/
    inline __device__ void set(uint pi, const real4 &totalForce){
      newForce[pi] += make_real4(totalForce.x, totalForce.y, totalForce.z, real(0.0));
    }
    /*Initial value of the force, this is a trick to allow the template in transverseList
      to guess the type of my quantity, a real4 in this case. Just set it to the 0 value of 
    the type of your quantity (0.0f for a real i.e)*/
    inline __device__ real4 zero(){
      return make_real4(real(0.0));
    }

    inline __device__ ParticleInfo getInfo(uint index){      
      return {tex1Dfetch<real4>(params.texSortPos, index)};
    }
 
  private:
    real4* newForce;
  };


  
  //CPU kernel caller
  void computePairForce(real4 *sortPos, real4 *force,
   			uint *cellStart, uint *cellEnd,
   			uint *particleIndex, 
   			uint N, uint ntypes){
    /*An instance of the class that holds the function that computes the force*/
    if(ntypes>1){ /*ntypes is templated, really makes a difference*/
      forceTransverser<true> ft(force); //It needs the addres of the force in device memory
      /*Transverse the neighbour list for each particle, using ft to compute the force in each pair*/
      transverseListD<<<GPU_Nblocks, GPU_Nthreads>>>(
     //transverseListDtpp<<<GPU_Nblocks*tpp, GPU_Nthreads>>>(
						     ft,
						     particleIndex, cellStart, cellEnd,
						     N);

    }
    else{
      forceTransverser<false> ft(force); //It needs the addres of the force in device memory
      /*Transverse the neighbour list for each particle, using ft to compute the force in each pair*/
      transverseListD<<<GPU_Nblocks, GPU_Nthreads>>>(
     //transverseListDtpp<<<GPU_Nblocks*tpp, GPU_Nthreads>>>(
						     ft,
						     particleIndex, cellStart, cellEnd,
						     N);
    }

    //cudaCheckErrors("computeForce");
  }

  /****************************ENERGY***************************************/

  /*This class is analogous to forceTransverser, see for reference*/
  //tags: energy compute energyij
  class energyTransverser{
  public:
    struct ParticleInfo{
      real4 pos;
    };

    energyTransverser(real *Energy):Energy(Energy){ };
    /*Returns the energy between two positions*/
    inline __device__ real compute(const ParticleInfo &R1,const ParticleInfo &R2){
      real3 r12 = make_real3(R2.pos-R1.pos);

      apply_pbc(r12);

      real r2 = dot(r12,r12);
      /*Squared distance between 0 and 1*/
      float r2c = r2*params.invrc2_pot;
      real sigma2= real(1.0);
      real epsilon = real(1.0); 
      if(params.ntypes>1){
	uint ti= (uint)(R1.pos.w+0.5);
	uint tj= (uint)(R2.pos.w+0.5);
	real2 pot_params = params.potParams[ti+tj*params.ntypes];
	sigma2 = pot_params.x*pot_params.x;
	epsilon = pot_params.y;

	r2c /= sigma2;
      }

#if __CUDA_ARCH__>210      
      float E = epsilon*tex1D<float>(params.texEnergy.tex, r2c);
#else
      float E = epsilon*tex1D(texEnergy, r2c);
#endif
      
      return E;
    }
    inline __device__ void set(uint pi, real energy){
      Energy[pi] = energy;
    }
    inline __device__ real zero(){
      return real(0.0);
    }
    inline __device__ ParticleInfo getInfo(uint index){      
      return {tex1Dfetch<real4>(params.texSortPos, index)};
    }

  private:
    real *Energy;
  };

  
  real computePairEnergy(real4 *sortPos, real *energy,
			 uint *cellStart, uint *cellEnd,
			 uint *particleIndex, 
			 uint N){

    /*Analogous to computeForce, see for reference*/
    energyTransverser et(energy);
    transverseListD<<<GPU_Nblocks, GPU_Nthreads>>>(et,
						   particleIndex, cellStart, cellEnd,
						   N);
    device_ptr<real> d_e(energy);
    real sum;
    sum = thrust::reduce(d_e, d_e+N, 0.0f);
    //Returns energy per particle*/
    return (sum/(real)N);
    //cudaCheckErrors("computeForce");
  }


  /****************************VIRIAL***************************************/
  /*Analogous to forceTransverser, see for reference*/
  //tags: virial compute virialij
  class virialTransverser{
  public:
    struct ParticleInfo{
      real4 pos;
    };

    virialTransverser(real *virial):Virial(virial){ };
    inline __device__ real compute(const ParticleInfo &R1,const ParticleInfo &R2){
      real3 r12 = make_real3(R2.pos-R1.pos);
      apply_pbc(r12);

      real r2 = dot(r12,r12);
      /*Squared distance between 0 and 1*/
      float r2c = r2*params.invrc2_pot;
      real sigma2= real(1.0);
      real epsilon = real(1.0); 
      if(params.ntypes>1){
	uint ti= (uint)(R1.pos.w+0.5);
	uint tj= (uint)(R2.pos.w+0.5);
	real2 pot_params = params.potParams[ti+tj*params.ntypes];
	sigma2 = pot_params.x*pot_params.x;
	epsilon = pot_params.y;

	r2c /= sigma2;
      }

#if __CUDA_ARCH__>210
      real fmod = epsilon* (real) tex1D<float>(params.texForce.tex, r2c);
#else /*Use texture reference*/
      real fmod = epsilon* (real) tex1D(texForce, r2c);
#endif

      // P = rhoKT + (1/2dV)sum_ij( Fij·rij ) //Compute only the Fij·rij, the rest is done outside
      return dot(fmod*r12,r12);
    }
    inline __device__ void set(uint pi, real virial){
      Virial[pi] = virial;
    }
    inline __device__ real zero(){
      return 0.0f;
    }

    inline __device__ ParticleInfo getInfo(uint index){
      return {tex1Dfetch<real4>(params.texSortPos, index)};
    }

  private:
    real *Virial;
  };




  //CPU kernel caller
  real computePairVirial(real4 *sortPos, real *virial,
			  uint *cellStart, uint *cellEnd,
			  uint *particleIndex, 
			  uint N){

    virialTransverser ft(virial);
    transverseListD<<<GPU_Nblocks, GPU_Nthreads>>>(ft,
						   particleIndex, cellStart, cellEnd,
						   N);
    device_ptr<real> d_vir(virial);
    real sum;
    // P = rhoKT + (1/2dV)sum_ij( Fij·rij ) This function returns (1/2)sum_ij( Fij·rij )
    sum = thrust::reduce(d_vir, d_vir+N, 0.0f);
    return (sum*real(0.5));
    //cudaCheckErrors("computeForce");
  }



  /*******************************************DPD********************************************/

  /**********************FORCE********************/


  //Random number, the seed is used to recover a certain number in the random stream
  //TODO: This is a bit awkward, probably it will be best to manually generate the number
  inline __device__ real randGPU(const ullint &seed, curandState *rng){
    curand_init(seed, 0, 0, rng);
    #if defined SINGLE_PRECISION
    return curand_normal(rng);
    #else
    return curand_normal_double(rng);
    #endif
  }

  /*Very similar to forceTransverser, but now the force depends on the velocity aswell!*/
  //tags: forceijDPD forceDPDij
  class forceDPDTransverser{
  public:
    /*A struct with all the information the force compute needs*/
    struct ParticleInfo{      
      real4 pos; //Pos has to be the first element always!
      uint pi;
    };

    /*I need the device pointer to force*/
    forceDPDTransverser(real4 *newForce):newForce(newForce){};
    
    /*Compute the force between two particles*/
    /*As in all this file, the __device__ function has access to all the parameters in params and,
      in this case, paramsDPD. See PairForcesGPU.cuh*/
    
    inline __device__ real4 compute(const ParticleInfo &R1,const ParticleInfo &R2){
      real3 r12 = make_real3(R1.pos-R2.pos);  
      apply_pbc(r12);

      real r2 = dot(r12,r12);
      /*Squared distance between 0 and 1*/
      real r2c = r2*params.invrc2_pot;

      real sigma2= real(1.0);
      real epsilon = real(1.0); 
      if(params.ntypes>1){
	uint ti= (uint)(R1.pos.w+0.5);
	uint tj= (uint)(R2.pos.w+0.5);
	real2 pot_params = params.potParams[ti+tj*params.ntypes];
	sigma2 = pot_params.x*pot_params.x;
	epsilon = pot_params.y;

	r2c /= sigma2;
      }

      
      real w = real(0.0); //The intensity of the DPD thermostat 
      real rinv = real(0.0);
      if(r2c<real(1.0)){
	if(r2c==real(0.0)) return make_real4(real(0.0));
	//w = r-rc -> linear
	rinv = rsqrt(r2);
	w = rinv-params.invrc_pot;
      }
      else return make_real4(real(0.0));
      


      uint i0 = R1.pi;
      uint j0 = R2.pi;

      real4 V1=tex1Dfetch<real4>(paramsDPD.texSortVel, i0);
      real4 V2=tex1Dfetch<real4>(paramsDPD.texSortVel, j0);
      
      real3 v12 = make_real3(V1-V2);      
      /*Prepare the seed for the RNG, it must be the same seed
	for pair ij and ji!*/
      if(i0>j0)
	swap(i0,j0);
      
      curandState rng;
      real randij = randGPU(i0+(ullint)gcnfGPU.N*j0 + seedGPU, &rng);

      //fmod = paramsDPD.A*w; //Soft force
#if __CUDA_ARCH__>210
      real fmod = -epsilon* (real) tex1D<float>(params.texForce.tex, r2c);
#else /*Use texture reference*/
      real fmod = -epsilon* (real) tex1D(texForce, r2c);
#endif
      
      fmod -= paramsDPD.gamma*w*w*dot(r12,v12); //Damping
      fmod += paramsDPD.noiseAmp*randij*w; //Random force
      return make_real4((real)fmod*r12);
    }
    /*Update the force acting on particle pi, pi is in the normal order*/
    inline __device__ void set(uint pi, const real4 &totalForce){
      newForce[pi] += totalForce;
    }
    /*Initial value of the force, this is a trick to allow the template in transverseList
      to guess the type of my quantity, a real4 in this case. Just set it to the 0 value of 
    the type of your quantity (0.0f for a real i.e)*/
    inline __device__ real4 zero(){
      return make_real4(real(0.0));
    }

    inline __device__ ParticleInfo getInfo(uint index){
      return {tex1Dfetch<real4>(params.texSortPos, index),
	      index};
    }
  private:
    real4* newForce;
  };
  
  //CPU kernel caller
  void computePairForceDPD(real4 *force,
			   uint *particleIndex,
			   uint *cellStart,
			   uint *cellEnd,
			   uint N, ullint seed){
    //TODO: This should not be needed somehow
    gpuErrchk(cudaMemcpyToSymbol(seedGPU, &seed, sizeof(seedGPU)));
    
    forceDPDTransverser ft(force);
    
    transverseListD<<<GPU_Nblocks, GPU_Nthreads>>>(ft, 
     						   particleIndex , cellStart, cellEnd,
     						   N);
    //cudaCheckErrors("computeForce");
  }


  

  __global__ void needsUpdateGPUD(real4 *pos, real4 *old_pos, real threshold, int *flagGPU, uint N){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id>=N) return;
    real3 rij = make_real3(pos[id])-make_real3(old_pos[id]);
    if(dot(rij, rij) > threshold*threshold)
      atomicMax(flagGPU, id); 
  }  


  bool needsUpdateGPU(real4 *pos, real4 *old_pos, real threshold, uint N){
    static int *flagGPU = NULL;
    if(!flagGPU) cudaMalloc(&flagGPU, sizeof(int));

    cudaMemset(flagGPU, 0xffffffff, sizeof(int));
    needsUpdateGPUD<<<GPU_Nblocks, GPU_Nthreads>>>(pos, old_pos, threshold, flagGPU, N);
  
    int flag;
    cudaMemcpy(&flag, flagGPU, sizeof(int), cudaMemcpyDeviceToHost);   
    return flag!=0xffffffff;

  }

  void reorderPosGPU( real4 *sortPos, real4 *pos, uint *particleIndex, uint N){
    reorderPosD<<<GPU_Nblocks, GPU_Nthreads>>>(sortPos, pos, particleIndex, N);
  }

  
}
  

/*Raul P. Pelaez 2021. NBody base submodule. Does not need a particleData or particleGroup

  An NBody is a very lightweight object than can be used to process Transversers with an all-with-all nbody interaction O(N^2).

USAGE:

Create an instance:
NBody nbody;

Use it to process a transverser:

nbody.transverse(particle_list, index_list, myTransverser, numberParticles, cudaStream);

Where particle_list is a random access iterator of size numberParticles.
The type of the elements in particle_list can be any POD type.
If it is a list of real4* the standard Transverser interface can be used, otherwise the first two arguments of the compute function will require to have the type of the elements in particle_list ( std::iterator_traits<decltype(particle_list)>::value_type ).

index_list is an iterator containing integer types that will be used as indices for the particle_list and transverser. This can be used to work only on a subset of the particles. If not present all particles will be processed.

See more about transversers and how to implement them in the wiki page[1].


[1] https://github.com/RaulPPelaez/UAMMD/wiki/Transverser
*/
#ifndef NBODYBASE_CUH
#define NBODYBASE_CUH
#include"global/defines.h"
#include"third_party/type_names.h"
#include"utils/cxx_utils.h"
#include"utils/TransverserUtils.cuh"
#include "utils/debugTools.h"

namespace uammd{

  namespace nbody_ns{

    /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
    template<class Transverser, class Iterator, class IndexIterator>
    __global__ void transverseGPU(const Iterator particle_list,
				  IndexIterator threadId2Index,
				  int numTiles, /*Thread paralellism level,
						  controls how many elements are stored in
						  shared memory and
						  computed in parallel between synchronizations*/
				  Transverser tr, int N,
				  size_t sharedInfoOffset,
				  //Requested shared memory of Transverser in bytes
				  size_t transverserShMemSize){
      const int tid = blockIdx.x*blockDim.x+threadIdx.x;
      using PerParticleValue = typename std::iterator_traits<Iterator>::value_type;
      //All threads must pass through __syncthreads,
      //but when N is not a multiple of 32 some threads are assigned a particle i>N.
      //This threads cant return, so they are masked to not do any work
      bool active = true;
      if(tid>=N) active = false;
      const int id = threadId2Index[tid];
      using Adaptor = SFINAE::TransverserAdaptor<Transverser>;
      //Each thread handles the interaction between particle id and all the others
      //Storing blockDim.x values in shared memory and processing all of them in parallel
      SFINAE::TransverserAdaptor<Transverser> adaptor;
      extern __shared__ char shMem[];
      PerParticleValue *shValue =  (PerParticleValue*)(shMem + transverserShMemSize);
      void *shInfo =  (void*)(shMem + sharedInfoOffset + transverserShMemSize);
      PerParticleValue valuei;
      if(active) {
	valuei = particle_list[id]; //My particle's value
	//Get additional info if needed
	adaptor.getInfo(tr, id);
      }
      //Get the initial value of the quantity to compute, i.e force
      auto quantity = Adaptor::zero(tr);
      //Distribute the N particles in numTiles tiles. Storing in each tile blockDim.x values in shared memory
      //This way all threads are accesing the same memory addresses at the same time
      for(int tile = 0; tile<numTiles; tile++){
	//Load this tiles particles values to shared memory
	int i_load = tile*blockDim.x+threadIdx.x;	
	if(i_load<N){ //Even if im not active, my thread may load a value each tile to shared memory.
	  i_load = threadId2Index[i_load];
	  shValue[threadIdx.x] = particle_list[i_load];
	  Adaptor::fillSharedMemory(tr, shInfo, i_load);
	}
	//Wait for all threads to arrive
	__syncthreads();
	//Go through all the particles in the current tile
#pragma unroll 8
	for(uint counter = 0; counter<blockDim.x; counter++){
	  if(!active) break; //An out of bounds thread must be masked
	  int cur_j = tile*blockDim.x+counter;
	  if(cur_j<N){//If the current particle exists, compute and accumulate
	    //Compute and accumulate the current pair using:
	    //  -values for i and j
	    //  -inofi and infoj (handled by Delegator)
	    Adaptor::accumulate(tr, quantity, adaptor.computeSharedMem(tr, valuei, shValue[counter], shInfo, counter));
	  }
	}//End of particles in tile loop
	__syncthreads();
      }//End of tile loop
      //Write the result to global memory
      if(active)
	tr.set(id, quantity);
    }
  }

  class NBodyBase{
  public:
    NBodyBase(){
      System::log<System::DEBUG>("[NBody] Created");
    }

    //For each particle in the list (only in the elements given by the argument indices), applies the transverser to every other.
    template<class Iterator, class IndexIterator, class Transverser>
    static void transverse(Iterator particle_list, IndexIterator &indices,
			   Transverser &a_tr,
			   int numberParticles, cudaStream_t st = 0){
      using PerParticleValue = typename std::iterator_traits<Iterator>::value_type;
      System::log<System::DEBUG2>("[NBody] Transversing with %s", type_name<Transverser>().c_str());
      int Nthreads = 128<numberParticles?128:numberParticles;
      int Nblocks  = (numberParticles+Nthreads-1)/Nthreads;
      int numtiles = (numberParticles + Nthreads-1)/Nthreads;
      //NBody will store the transverser's info in shared memory
      constexpr size_t info_size = SFINAE::Delegator<Transverser>::sizeofInfo();
      constexpr size_t info_align = alignof(typename SFINAE::Delegator<Transverser>::InfoType);
      constexpr size_t sh_size = info_size+sizeof(PerParticleValue);
      size_t extra_alignment = (sizeof(PerParticleValue))%info_align;
      //Get the transverser's shared memory needs (if any)
      size_t extra_sh = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(a_tr);
      System::log<System::DEBUG4>("[NBody] %zu particles", numberParticles);
      System::log<System::DEBUG4>("[NBody] %zu info_sh", info_size);
      System::log<System::DEBUG4>("[NBody] %zu info_align", info_align);
      System::log<System::DEBUG4>("[NBody] %zu extra_alignment", extra_alignment);
      System::log<System::DEBUG4>("[NBody] %zu sh mem", Nthreads*sh_size+extra_alignment);
      System::log<System::DEBUG4>("[NBody] %zu Additional sh mem", extra_sh);   
      nbody_ns::transverseGPU<<<Nblocks, Nthreads, Nthreads*sh_size+extra_sh+extra_alignment, st>>>(particle_list,
										    indices,
										    numtiles,
										    a_tr,
										    numberParticles,
										    extra_alignment + Nthreads*sizeof(PerParticleValue),
										    extra_sh);
      CudaCheckError();
    }

    //For each particle in the list, applies the transverser to every other.
    template<class Iterator, class Transverser>
    static inline void transverse(Iterator &particle_list, Transverser &a_tr, int numberParticles, cudaStream_t st = 0){
      auto cit = thrust::make_counting_iterator<int>(0);
      transverse(particle_list, cit, numberParticles, st);
    }
  };

}


#endif

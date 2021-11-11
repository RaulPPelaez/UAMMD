/*Raul P. Pelaez 2017. NBody submodule.

  An NBody is a very lightweight object than can be used to process Transversers with an all-with-all nbody interaction O(N^2).

USAGE:

Create an instance:
NBody nbody(particleData, particleGroup, system);

Use it to process a transverser:

nbody.transverse(myTransverser, cudaStream);


It has a very low memory footprint and a very fast initialization,
  so do not bother storing it, just create it when needed.

See more about transversers and how to implement them at the end of this file.

EXAMPLES:

You have examples of the usage of Nbody in NBodyForces.cuh, PairForces.cuh and BDHI_Lanczos.cu.
*/
#ifndef NBODY_CUH
#define NBODY_CUH
#include"ParticleData/ParticleGroup.cuh"

#include"global/defines.h"
#include"third_party/type_names.h"

#include"utils/cxx_utils.h"
#include"utils/TransverserUtils.cuh"
#include "utils/debugTools.h"

namespace uammd{

  namespace NBody_ns{

    /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
    template<class Transverser, class Iterator>
    __global__ void transverseGPU(const real4* pos,
				  Iterator groupIterator,
				  int numTiles, /*Thread paralellism level,
						  controls how many elements are stored in
						  shared memory and
						  computed in parallel between synchronizations*/
				  Transverser tr, int N,
				  size_t sharedInfoOffset,
				  //Requested shared memory of Transverser in bytes
				  size_t transverserShMemSize){
      const int tid = blockIdx.x*blockDim.x+threadIdx.x;
      //All threads must pass through __syncthreads,
      //but when N is not a multiple of 32 some threads are assigned a particle i>N.
      //This threads cant return, so they are masked to not do any work
      bool active = true;
      if(tid>=N) active = false;
      const int id = groupIterator[tid];
      using Adaptor = SFINAE::TransverserAdaptor<Transverser>;
      //Each thread handles the interaction between particle id and all the others
      //Storing blockDim.x positions in shared memory and processing all of them in parallel
      SFINAE::TransverserAdaptor<Transverser> adaptor;
      extern __shared__ char shMem[];
      real4 *shPos =  (real4*)(shMem + transverserShMemSize);
      void *shInfo =  (void*)(shMem + sharedInfoOffset + transverserShMemSize);
      real4 pi;
      if(active) {
	pi = pos[id]; //My position
	//Get additional info if needed
	adaptor.getInfo(tr, id);
      }
      //Get the initial value of the quantity to compute, i.e force
      auto quantity = Adaptor::zero(tr);
      //Distribute the N particles in numTiles tiles. Storing in each tile blockDim.x positions in shared memory
      //This way all threads are accesing the same memory addresses at the same time
      for(int tile = 0; tile<numTiles; tile++){
	//Load this tiles particles positions to shared memory
	int i_load = tile*blockDim.x+threadIdx.x;
	if(i_load<N){ //Even if im not active, my thread may load a position each tile to shared memory.
	  i_load = groupIterator[i_load];
	  shPos[threadIdx.x] = pos[i_load];
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
	    //  -positions
	    //  -inofi and infoj (handled by Delegator)
	    Adaptor::accumulate(tr, quantity, adaptor.computeSharedMem(tr, pi, shPos[counter], shInfo, counter));
	    //Adaptor::accumulate(tr, quantity, adaptor.compute(tr, cur_j, pi, shPos[counter]));
	  }
	}//End of particles in tile loop
	__syncthreads();
      }//End of tile loop
      //Write the result to global memory
      if(active)
	tr.set(id, quantity);
    }

  }

  class NBody{
    shared_ptr<ParticleGroup> pg;
    shared_ptr<ParticleData> pd;
    shared_ptr<System> sys;
  public:
    NBody(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys): pg(pg), pd(pd), sys(sys){
      sys->log<System::DEBUG>("[NBody] Created");
    }
    template<class Transverser>
    inline void transverse(Transverser &a_tr, cudaStream_t st = 0){
      sys->log<System::DEBUG2>("[NBody] Transversing with %s", type_name<Transverser>().c_str());
      int N = pg->getNumberParticles();
      int Nthreads = 128<N?128:N;
      int Nblocks  = (N+Nthreads-1)/Nthreads;
      int numtiles = (N + Nthreads-1)/Nthreads;
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      //NBody will store the transverser's info in shared memory
      constexpr size_t info_size = SFINAE::Delegator<Transverser>::sizeofInfo();
      constexpr size_t info_align = alignof(typename SFINAE::Delegator<Transverser>::InfoType);
      constexpr size_t sh_size = info_size+sizeof(real4);
      size_t extra_alignment = (sizeof(real4))%info_align;
      //Get the transverser's shared memory needs (if any)
      size_t extra_sh = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(a_tr);
      sys->log<System::DEBUG4>("[NBody] %zu particles", N);
      sys->log<System::DEBUG4>("[NBody] %zu info_sh", info_size);
      sys->log<System::DEBUG4>("[NBody] %zu info_align", info_align);
      sys->log<System::DEBUG4>("[NBody] %zu extra_alignment", extra_alignment);
      sys->log<System::DEBUG4>("[NBody] %zu sh mem", Nthreads*sh_size+extra_alignment);
      sys->log<System::DEBUG4>("[NBody] %zu Additional sh mem", extra_sh);   
      SFINAE::TransverserAdaptor<Transverser>::prepare(a_tr, pd);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      NBody_ns::transverseGPU<<<Nblocks, Nthreads, Nthreads*sh_size+extra_sh+extra_alignment, st>>>(pos.raw(),
										    groupIterator,
										    numtiles,
										    a_tr,
										    N,
										    extra_alignment + Nthreads*sizeof(real4),
										    extra_sh);
      CudaCheckError();
    }
  };

}


#endif

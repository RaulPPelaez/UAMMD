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

namespace uammd{

  namespace NBody_ns{

    /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
    template<class Transverser, class Iterator>
    __global__ void transverseGPU(const real4* __restrict__ pos,
				  Iterator groupIterator,
				  int numTiles, /*Thread paralellism level, 
						  controls how many elements are stored in 
						  shared memory and
						  computed in parallel between synchronizations*/
				  Transverser tr, uint N,
				  //Requested shared memory of Transverser in bytes
				  size_t transverserShMemSize);
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
      size_t info_size = SFINAE::Delegator<Transverser>::sizeofInfo()+sizeof(real4);
      
  
      //Get the transverser's shared memory needs (if any)
      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(a_tr);

      sys->log<System::DEBUG4>("[NBody] %d particles", N);
      sys->log<System::DEBUG4>("[NBody] %d sh mem", shMemorySize);
      
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      NBody_ns::transverseGPU<<<Nblocks, Nthreads, Nthreads*info_size+shMemorySize, st>>>(pos.raw(),
											  groupIterator,
											  numtiles,
											  a_tr,
											  N,
											  shMemorySize);
    }
  };


  namespace NBody_ns{

    /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
    template<class Transverser, class Iterator>
    __global__ void transverseGPU(const real4* __restrict__ pos,
				  Iterator groupIterator,
				  int numTiles, /*Thread paralellism level, 
						  controls how many elements are stored in 
						  shared memory and
						  computed in parallel between synchronizations*/
				  Transverser tr, uint N,
				  //Requested shared memory of Transverser in bytes
				  size_t transverserShMemSize){
      int tid = blockIdx.x*blockDim.x+threadIdx.x;
      /*All threads must pass through __syncthreads, 
	but when N is not a multiple of 32 some threads are assigned a particle i>N.
	This threads cant return, so they are masked to not do any work*/
      bool active = true;
      if(tid>=N) active = false;

      int id = groupIterator[tid];
    
      /*Each thread handles the interaction between particle id and all the others*/
      /*Storing blockDim.x positions in shared memory and processing all of them in parallel*/
      extern __shared__ char shMem[];

      real4 *shPos = (real4*) (shMem + transverserShMemSize);    
      void *shInfo = (void*) (shMem+blockDim.x*sizeof(real4) + transverserShMemSize);
    
      /*Delegator makes it possible to invoke this template even with a simple Transverser
	(one that doesnt have a getInfo method) by using a little SFINAE trick*/
      /*Note that in any case there is no overhead for calling Delegator, as the compiler
	is smart enough to just trash it all in any case*/
      SFINAE::Delegator<Transverser> del;
      real4 pi;    
      if(active) {
	pi = pos[id]; /*My position*/
	/*Get additional info if needed.
	  Note that this code is just trashed if Transverser is simple*/
	del.getInfo(tr, id);
      }
      /*Get the initial value of the quantity to compute, i.e force*/    
      auto quantity = tr.zero(); 
      /*Distribute the N particles in numTiles tiles.
	Storing in each tile blockDim.x positions in shared memory*/
      /*This way all threads are accesing the same memory addresses at the same time*/
      for(int tile = 0; tile<numTiles; tile++){
	/*Load this tiles particles positions to shared memory*/
	int i_load = tile*blockDim.x+threadIdx.x;
	if(i_load<N){ /*Even if im not active,
			my thread may load a position each tile to shared memory.*/
	  i_load = groupIterator[i_load];
	  shPos[threadIdx.x] = pos[i_load];
	  del.fillSharedMemory(tr, shInfo, i_load);
	}
	/*Wait for all threads to arrive*/
	__syncthreads();
	/*Go through all the particles in the current tile*/
#pragma unroll 8
	for(uint counter = 0; counter<blockDim.x; counter++){
	  if(!active) break; /*An out of bounds thread must be masked*/
	  int cur_j = tile*blockDim.x+counter; 
	  if(cur_j<N){/*If the current particle exists, compute and accumulate*/
	    /*Compute and accumulate the current pair using:
	      -positions
	      -inofi and infoj (handled by Delegator)
	    */
	    tr.accumulate(quantity, del.computeSharedMem(tr, pi, shPos[counter], shInfo, counter));
	  }
	}/*End of particles in tile loop*/
	__syncthreads();

      }/*End of tile loop*/
      /*Write the result to global memory*/
      if(active)
	tr.set(id, quantity);
    }
  
    
    //Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3
//     template<class Transverser, class Iterator>
//     __global__ void transverseGPUold(Iterator groupIterator, 
// 				     int numTiles, /*Thread paralellism level, 
// 						     controls how many elements are stored in 
// 						     shared memory and
// 						     computed in parallel between synchronizations*/
// 				     Transverser tr, uint N){
//       int tid = blockIdx.x*blockDim.x+threadIdx.x;
//       /*All threads must pass through __syncthreads, 
// 	but when N is not a multiple of 32 some threads are assigned a particle i>N.
// 	This threads cant return, so they are masked to not do any work*/
//       bool active = true;
//       if(tid>=N) active = false;

//       int id = groupIterator[tid];
//       /*Each thread handles the interaction between particle id and all the others*/
//       /*Storing blockDim.x positions in shared memory and processing all of them in parallel*/
//       using InfoType = decltype(tr.getInfo(id));
//       extern __shared__ char shMem[];    
//       InfoType *shInfo = (InfoType*) (shMem);
    
    
//       /*Delegator makes it possible to invoke this template even with a simple Transverser
// 	(one that doesnt have a getInfo method) by using a little SFINAE trick*/
//       /*Note that in any case there is no overhead for calling Delegator, as the compiler
// 	is smart enough to just trash it all in any case*/
//       InfoType infoi;
//       if(active) {
// 	/*Get necessary info if needed.
// 	  Note that this code is just trashed if Transverser is simple*/
// 	infoi = tr.getInfo(id);
//       }
//       /*Get the initial value of the quantity to compute, i.e force*/    
//       auto quantity = tr.zero(id); 
//       /*Distribute the N particles in numTiles tiles.
// 	Storing in each tile blockDim.x positions in shared memory*/
//       /*This way all threads are accesing the same memory addresses at the same time*/
//       for(int tile = 0; tile<numTiles; tile++){
// 	/*Load this tiles particles positions to shared memory*/
// 	const int i_load = tile*blockDim.x+threadIdx.x;
// 	if(i_load<N){ /*Even if im not active,
// 			my thread may load a position each tile to shared memory.*/
// 	  shInfo[threadIdx.x] = tr.getInfo(groupIterator[i_load]);
// 	}
// 	/*Wait for all threads to arrive*/
// 	__syncthreads();
// 	/*Go through all the particles in the current tile*/
// #pragma unroll 8
// 	for(uint counter = 0; counter<blockDim.x; counter++){
// 	  if(!active) break; /*An out of bounds thread must be masked*/
// 	  int cur_j = tile*blockDim.x+counter; 
// 	  if(cur_j<N){/*If the current particle exists, compute and accumulate*/
// 	    /*Compute and accumulate the current pair using:
// 	      -positions
// 	      -inofi and infoj (handled by Delegator)
// 	    */
// 	    tr.accumulate(quantity, tr.compute(infoi, shInfo[counter]));
// 	  }
// 	}/*End of particles in tile loop*/
// 	__syncthreads();

//       }/*End of tile loop*/
//       /*Write the result to global memory*/
//       if(active)
// 	tr.set(id, quantity);
//     }
  }




  /*An example of a Transverser*/
  namespace NBody_ns{


    /*This dumb transverser simply sums 1 for each particle each particle interacts with
      and stores the result in a device array called tmp*/
    /*After NbodyForces::sumForce is called, every element of tmp will contain N*/

    /*You can change computeType to whatever you want to compute, i.e a real4 for the force or an internally defined struct containing whatever and simply use this as a template*/
    struct SimpleCountingTransverser{
      typedef int computeType;
      typedef int infoType;
      /*Give me the device pointer!*/
      SimpleCountingTransverser(computeType* tmp): tmp(tmp){}
      /*Get any property arrays or do any pre compute computation, this will be called each step*/
      __host__ void prepare(shared_ptr<ParticleData> pd){
	cudaMemset(tmp, pd->getNumParticles()*sizeof(computeType), 0);
      }
      /*Start with 0*/
      inline __device__ computeType zero(int id){ return 0;}

      /*Is a general transverser, so additional info can be asked, probably the position and any other thing.*/
      /*i is the global index of a particle*/
      inline __device__ infoType getInfo(int i){return 1;}
      /*Just count the interaction*/
      inline __device__ computeType compute(const infoType &infoi, const infoType &infoj){
	/*With the positions two particles and the additional info, do something*/
	/*In this case, infoi and infoj are 1*/
	return infoj;
      }
      /*Sum the result of each interaction*/
      inline __device__ void accumulate(computeType &total, const computeType &cur){total += cur;}

      /*Write the final result to global memory*/
      inline __device__ void set(int id, const computeType &total){
	tmp[id] = total;
      }
    private:
      /*Any member here will be available in device memory,
	so you can include here parameters, arrays...
	and read them in the member __device__ functions*/
      /*In this case, only a device array is needed to store the results*/
      computeType* tmp;
    };
    
  }
  
}


#endif

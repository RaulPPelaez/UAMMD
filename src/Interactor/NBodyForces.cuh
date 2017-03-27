/*Raul P. Pelaez 2017. NBody Interactor module. All particles interact with all the others.
  
  See https://github.com/RaulPPelaez/UAMMD/wiki/NBody-Forces for more information

  NBody needs a transverser with the information of what to compute for each particle given all the others.
 You can see example Transversers at the end of this file. If your problem is very similar to one of this transversers, you can inherit from it or simply use it.

Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3
*/
#ifndef NBODY_CUH
#define NBODY_CUH

#include"Interactor.h"
#include"globals/defines.h"
#include"utils/SFINAE.cuh"

/*Template declaration of the kernel that will make the computation, declared below*/
namespace NBody_ns{
  template<class Transverser>
  __global__ void transverseGPUD(const __restrict__ real4  *pos,
				   __restrict__ real4 *force,
				   int numTiles, Transverser tr, uint N);
}


/*Nbody takes a transverser like NeighbourList,
  but this time all particles are neighbours of each other.
  See the wiki for more info on transversers. You can see an example below */
template<class Transverser>
class NBodyForces: public Interactor{
public:
  /*Some interactors can live in another cuda stream, computing in paralell*/
  NBodyForces(Transverser tr, cudaStream_t st = 0):Interactor(), tr(tr), st(st){
    /*Interactors are silent, 
      print_info is used to write any messages, variable values, etc.*/
    /*Small interactors like this, thought to be created and destroyed constantly 
      are meant to be very lightweithg both in memory and initialization time*/
    BLOCKSIZE= 1024;
    name = "NBody";
  }
  ~NBodyForces(){}
  
  void sumForce()  override{this->transverse(tr);} 
  real sumEnergy() override{return 0.0;}
  real sumVirial() override{return 0.0;}


  void print_info(){
    std::cerr<<"\t Transversing with: "<<typeid(Transverser).name()<<std::endl;
  }
  /*Use NBody with a different Transverser (of the same type)*/
  inline void transverse(const Transverser &a_tr); /*Implemented below*/
private:
  /*Beisdes what I have as an interactor, I only need a Transverser.
    Any additional info needed by the Transverser will be included in it*/
  Transverser tr;
  cudaStream_t st;
};


namespace NBody_ns{

  /*Reference: Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3*/
  template<class Transverser>
  __global__ void transverseGPU(const real4* __restrict__ pos,
				   int numTiles, /*Thread paralellism level, 
						   controls how many elements are stored in 
						   shared memory and
						   computed in parallel between synchronizations*/
			     Transverser tr, uint N){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    /*All threads must pass through __syncthreads, 
      but when N is not a multiple of 32 some threads are assigned a particle i>N.
      This threads cant return, so they are masked to not do any work*/
    bool active = true;
    if(id>=N) active = false;
    
    /*Each thread handles the interaction between particle id and all the others*/
    /*Storing blockDim.x positions in shared memory and processing all of them in parallel*/
    extern __shared__ char shMem[];

    real4 *shPos = (real4*) shMem;
    
    void *shInfo = (void*) (shMem+blockDim.x*sizeof(real4));
    
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
      const int i_load = tile*blockDim.x+threadIdx.x;
      if(i_load<N){ /*Even if im not active,
		     my thread may load a position each tile to shared memory.*/	
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
  
}


template<class Transverser>
void NBodyForces<Transverser>::transverse(const Transverser &a_tr){
  int Nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int Nblocks  = (N+Nthreads-1)/Nthreads;
  int numtiles = (N + Nthreads-1)/Nthreads;        
  
  /*TODO: all additional info cloud go to shared memory with a little SFINAE magic*/
  SFINAE::Delegator<Transverser> del;
  size_t info_size = del.sizeofInfo();  
  NBody_ns::transverseGPU<Transverser><<<Nblocks, Nthreads,
    Nthreads*(sizeof(real4)+info_size), st>>>(pos, numtiles, a_tr, N);
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
    /*Start with 0*/
    inline __device__ computeType zero(){ return 0;}

    /*Is a general transverser, so additional info can be asked, 
      note that you can simply not write this function and the transverser will be faster.
    In that case this would be a simple Transverser, and compute would only have two arguments, posi and posj*/
    inline __device__ infoType getInfo(int i){return 1;}
    /*Just count the interaction*/
    inline __device__ computeType compute(const real4 &pi, const real4 &pj,
		/*Only if getInfo exists*/const infoType &infoi, const infoType &infoj){
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


#endif
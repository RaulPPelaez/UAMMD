/*Raul P. Pelaez 2017. Verlet List implementation
  A neighbour list can compute and provide for each particle a list of particle indices that are closer than a certain distance to it.

  All neighbour list must be capable of:

  -Handling particle groups.
  -Provide an iterator with a list of neighbour particles for each particle
  -Provide an iterator with the number of neighbours of each particle.
  -Provide a "transverse" method that takes a Transverser and goes through every pair of particles with it (See NBodyForces.cuh or RadialPotential.cuh)


  The list for a certain particle i starts at particleOffset[i], after numberNeighbours[i], the neighbour list for particle i contains undefined data. i is the index of a particle referred to its group (particle id if the group contains all particles).


  VerletList uses CellList to construct a verlet list with a certain support increase beyond the cut off with the hope of having to construct the list only every few steps.
  Seems to work best for diluted systems. If the system is dense CellList works best as of now.


Usage:

   See PairForces.cu for a simple example on how to use a NL.
   Typically transverseList will do a much better job at using the list than asking for it and manually transversing it. But it could be useful if the list is to be used many times per step. 
 

TODO:
100- Make verletRadiusMultiplier self optimized somehow.
 */
#ifndef VERLETLIST_CUH
#define VERLETLIST_CUH

#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"utils/ParticleSorter.cuh"
#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include"utils/cxx_utils.h"
#include"utils/TransverserUtils.cuh"
#include"System/System.h"
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include"CellList.cuh"

#include<limits>

namespace uammd{
  namespace VerletList_ns{

    /*Store current positions in lastPos, in id order*/
    template<class GroupIterator>
    __global__ void updateLastPos(real4 *pos,
				  real4 *lastPos,
				  GroupIterator groupIndexes,		
				  int numberParticles){

      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=numberParticles) return;
      int gid = groupIndexes[id];
      lastPos[id] = pos[gid];      
    }


    /*Check if a particle has moved farther than maxDist since last list rebuild*/
    template<class GroupIterator>
    __global__ void checkLastPos(real4 *pos,
				 real4 *lastPos,
				 GroupIterator groupIndexes,
				 real maxDist,
				 uint* updateFlag,
				 Box box,
				 int numberParticles){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=numberParticles) return;


      int gid = groupIndexes[id];
      real3 currentPos = make_real3(pos[gid]);
      real3 prevPos = make_real3(lastPos[id]);

      real3 rij = box.apply_pbc(currentPos-prevPos);
      if(dot(rij, rij)>=(maxDist*maxDist)){
	atomicAdd(updateFlag, 1u);
      }
    }   
    
  }
  class VerletList{
  protected:
    shared_ptr<ParticleData> pd; 
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;


    shared_ptr<CellList> cl;

    real verletRadiusMultiplier;
    real currentCutOff;
    Box currentBox;

    bool force_next_update = true;
    
    connection reorderConnection;
    
    thrust::device_vector<real4> lastPos;
    uint *updateFlagGPU;
  public:
        
    VerletList(shared_ptr<ParticleData> pd,
	       shared_ptr<System> sys):
      VerletList(pd, std::make_shared<ParticleGroup>(pd, sys), sys){ }

    VerletList(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys): pd(pd), pg(pg), sys(sys){
      sys->log<System::MESSAGE>("[VerletList] Created");

      cl = std::make_shared<CellList>(pd, pg, sys);
      reorderConnection = pd->getReorderSignal()->connect([this](){this->handle_reorder();});
      CudaSafeCall(cudaMalloc(&updateFlagGPU, sizeof(uint)));
      uint zero = 0;
      CudaSafeCall(cudaMemcpy(updateFlagGPU, &zero, sizeof(uint), cudaMemcpyHostToDevice));
      verletRadiusMultiplier = 1.1;
      CudaCheckError();
    }
    ~VerletList(){
      sys->log<System::DEBUG>("[VerletList] Destroyed");
      reorderConnection.disconnect();
      CudaSafeCall(cudaFree(updateFlagGPU));
      CudaCheckError();
    }

    //Update the verlet list if necessary and return it
    //NeighbourListData::particleStride will provide particle indexes inside the group, to get particle id (aka index in the global array) use pg->getIndexIterator();
    CellList::NeighbourListData getNeighbourList(cudaStream_t st = 0){
      return cl->getNeighbourList(st);
    }
    
    template<class Transverser>
    void transverseList(Transverser &tr, cudaStream_t st = 0){
      sys->log<System::DEBUG2>("[VerletList] Transversing Verlet list with %s", type_name<Transverser>().c_str());
      cl->transverseListWithNeighbourList(tr, st);
    }

  private:
    void updateLastPos(cudaStream_t st = 0){

      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto groupIndexesIterator  = pg->getIndexIterator(access::location::gpu);      

      try{
	lastPos.resize(numberParticles);
      }
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[VerletList] Thrust could not resize lastPos with error: %s", e.what());
      }

      VerletList_ns::updateLastPos<<<numberParticles/128+1, 128, 0, st>>>(pos.raw(),
									  thrust::raw_pointer_cast(lastPos.data()),
									  groupIndexesIterator,
									  numberParticles);
      CudaCheckError();

    }
  public:
    bool needsRebuild(Box box, real cutOff, cudaStream_t st = 0){
      CudaCheckError();
      sys->log<System::DEBUG2>("[VerletList] Checking for rebuild");
      pd->hintSortByHash(box, make_real3(cutOff*verletRadiusMultiplier));
      
      if(force_next_update){
	force_next_update = false;
	currentCutOff = cutOff;
	currentBox = box;
	sys->log<System::DEBUG3>("[VerletList] Forced list rebuild");
	this->updateLastPos();
	return true;
      }

      if(cutOff != currentCutOff){
	sys->log<System::DEBUG3>("[VerletList] cutOff changed from %e to %e. Rebuilding list.",
				 currentCutOff, cutOff);
	currentCutOff = cutOff;
	currentBox = box;
	this->updateLastPos();
	return true;
      }
      if(box.boxSize.x != currentBox.boxSize.x or
	 box.boxSize.y != currentBox.boxSize.y or
	 box.boxSize.z != currentBox.boxSize.z){
	sys->log<System::DEBUG3>("[VerletList] Box size changed, rebuilding list.");
	currentBox = box;
	this->updateLastPos();
	return true;
      }


      int numberParticles = pg->getNumberParticles();
      if(lastPos.size() != numberParticles){
	sys->log<System::DEBUG3>("[VerletList] Number particles changed, forcing rebuild");
	this->updateLastPos();
	return true;
      }
      real thresholdDistance = (verletRadiusMultiplier*currentCutOff-currentCutOff)/2.0;
      {
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto groupIndexesIterator  = pg->getIndexIterator(access::location::gpu);      
	sys->log<System::DEBUG3>("[VerletList] lastPos size: %d", lastPos.size());
	
	VerletList_ns::checkLastPos<<<numberParticles/128+1, 128, 0, st>>>(pos.raw(),
									   thrust::raw_pointer_cast(lastPos.data()),
									   groupIndexesIterator,
									   thresholdDistance,
									   updateFlagGPU,
									   currentBox,
									   numberParticles);
      }
      CudaCheckError();
      int updateFlag=0;
      CudaSafeCall(cudaMemcpy(&updateFlag, updateFlagGPU, sizeof(uint), cudaMemcpyDeviceToHost));
      sys->log<System::DEBUG3>("[VerletList] updateFlag: %d", updateFlag);
      if(updateFlag>0){
	int zero = 0;
	CudaSafeCall(cudaMemcpy(updateFlagGPU, &zero, sizeof(uint), cudaMemcpyHostToDevice));
	sys->log<System::DEBUG3>("[VerletList] %d particles moved beyond the threshold (%e), forcing rebuild.",
				 updateFlag, thresholdDistance);	
	this->updateLastPos();
	return true;
      }
      else{
	sys->log<System::DEBUG3>("[VerletList] No need for a rebuild");
	return false;
      }
    }
    
    void updateNeighbourList(Box box, real cutOff, cudaStream_t st = 0){
      if(this->needsRebuild(box, cutOff, st) == false) return;
      sys->log<System::DEBUG3>("[VerletList] Updating list");
      cl->updateNeighbourList(currentBox, currentCutOff*verletRadiusMultiplier, st);
    }
  protected:
    void handle_reorder(){
      sys->log<System::DEBUG3>("[VerletList] Particles sorted, forcing next rebuild.");
      force_next_update = true;
    }
  };


  }
#endif




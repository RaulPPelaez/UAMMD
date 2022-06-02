/*Raul P. Pelaez 2020. Verlet List UAMMD interface
  A NeighbourList can compute and provide for each particle a list of particle indices that are closer than a certain distance to it.

  There are three ways to interact with NeighbourList:

  1. Providing a Transverser to it
  2. Asking for a NeighbourContainer from it
  3. Any additional mechanism own to the particular instance (a.i. you can get the cell list binning structure with CellList)

  Methods 1. and 2. are tipically the fastest ones and ensure the code will work for any UAMMD neighbour list.

  See usage for instructions on how to use each method.

  All neighbour list must be capable of:

  -Handling particle groups.
  -Provide an iterator with a list of neighbour particles for each particle*
  -Provide a "transverse" method that takes a Transverser and goes through every pair of particles with it (See NBodyForces.cuh or RadialPotential.cuh)
  -Provide a NeighbourContainer able to provide, for each particle, a forward iterator with its neighbours.*

  *Note that this does not imply that a neighbour list is constructed, just that the iterator is able to provide the neighbours somehow "as is".


USAGE:

 //Create a VerletList:
 auto nl = make_shared<VerletList>(pd, pg, sys);
 //Update a list
 //If you pass a grid, the cellSize should be >= cutoff
 nl->update([A box or a grid], cutOff);

 //THE DIFFERENT INTERFACE METHODS
 
 //Set the cut off multiplier (verlet radius = rcut*multiplier)
 nl->setCutOffMultiplier(multiplier);

 //Traverse the list using the internal CellList mechanism(Method 1)
 nl->transverseList([A transverser],cudaStream);

 //Get a NeighbourContainer (Method 2.). This is equivalent to Method 1 (see transverseList kernel), but instead of having to provide a Transverser you can use this structure to get iterators to the neighbours of each particle (as the aforementioned kernel does) and process them manually. This allows for more versatility.
 //See examples/NeighbourListIterator.cu for an example
 auto nc = cl->getNeighbourContainer();

 //Get a Verlet list to use manually, which provides a spatial binning and a list with the bin of each particle (Method 3.).
 auto verlet_list_data = cl->getVerletList();

For using the Verlet list manually using getVerletList:
  The list for a certain particle i starts at particleOffset[i], after numberNeighbours[i], the neighbour list for particle i contains undefined data. i is the index of a particle referred to its group (particle id if the group contains all particles).

See PairForces.cu or examples/NeighbourListIterator.cu for examples on how to use a NL.


*/
#ifndef VERLETLIST_CUH
#define VERLETLIST_CUH
#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include"utils/TransverserUtils.cuh"
#include<thrust/device_vector.h>
#include"VerletList/VerletListBase.cuh"
#include"VerletList/NeighbourContainer.cuh"
#include"Interactor/NeighbourList/common.cuh"
namespace uammd{

  class VerletList{
  protected:
    shared_ptr<ParticleGroup> pg;
    VerletListBase nl;
    bool forceNextUpdate = true;
    connection posWriteConnection, reorderConnection;
    Box currentBox;
    real currentCutOff;

  public:

    VerletList(shared_ptr<ParticleData> pd):
      VerletList(std::make_shared<ParticleGroup>(pd, "All")){}

    VerletList(shared_ptr<ParticleGroup> pg): pg(pg){
      System::log<System::MESSAGE>("[VerletList] Created");
      auto pd = pg->getParticleData();
      posWriteConnection = pd->getPosWriteRequestedSignal()->connect([this](){this->handlePosWriteRequested();});
      reorderConnection = pd->getReorderSignal()->connect([this](){this->handleReorder();});
      CudaCheckError();
    }

    ~VerletList(){
      System::log<System::DEBUG>("[VerletList] Destroyed");
      posWriteConnection.disconnect();
      reorderConnection.disconnect();
    }

    void update(Box box, real cutOff, cudaStream_t st = 0){
      if(needsRebuild(box, cutOff)){
	System::log<System::DEBUG1>("[VerletList] Updating verlet list.");
	auto pd = pg->getParticleData();
	pd->hintSortByHash(box, make_real3(cutOff*0.5));
	currentBox = box;
	currentCutOff = cutOff;
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto posIter = pg->getPropertyIterator(pos);
	int numberParticles = pg->getNumberParticles();
	nl.update(posIter, numberParticles, box, cutOff, st);
      }
    }

    void update(Grid in_grid, real3 cutOff, cudaStream_t st = 0){
      update(in_grid.box, make_real3(cutOff), st);
    }

    void update(Box box, real3 cutOff, cudaStream_t st = 0){
      if(cutOff.x != cutOff.y or cutOff.x != cutOff.z){
	System::log<System::ERROR>("[VerletList] Cannot work with a different cut off in each direction");
	throw std::runtime_error("[VerletList] Invalid argument");
      }
      real rcut = cutOff.x;
      update(box, rcut, st);
    }

    template<class Transverser>
    void transverseList(Transverser &tr, cudaStream_t st = 0){
      System::log<System::DEBUG2>("[VerletList] Transversing Neighbour List with %s", type_name<Transverser>().c_str());
      const int numberParticles = pg->getNumberParticles();
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      auto globalIndex = pg->getIndexIterator(access::location::gpu);
      SFINAE::TransverserAdaptor<Transverser>::prepare(tr, pg->getParticleData());
      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);
      NeighbourList_ns::transverseWithNeighbourContainer<<<Nblocks, Nthreads, shMemorySize, st>>>(tr,
									       globalIndex,
									       this->getNeighbourContainer(),
									       numberParticles);
      CudaCheckError();
    }

    VerletListBase::VerletListData getVerletList(){
      return nl.getVerletList();
    }

    VerletListBase_ns::NeighbourContainer getNeighbourContainer(){
      auto listData = this->getVerletList();
      return VerletListBase_ns::NeighbourContainer(listData);
    }

    void setCutOffMultiplier(real newMultiplier){
      forceNextUpdate = true;
      nl.setCutOffMultiplier(newMultiplier);
    }

    int getNumberOfStepsSinceLastUpdate(){
      return nl.getNumberOfStepsSinceLastUpdate();
    }

  private:
    void handlePosWriteRequested(){
      System::log<System::DEBUG1>("[VerletList] Issuing a list update after positions were written to.");
      forceNextUpdate = true;
    }

    void handleReorder(){
      System::log<System::DEBUG1>("[VerletList] Issuing a list update after a reorder.");
      forceNextUpdate = true;
      nl.forceNextUpdate();
    }

    bool needsRebuild(Box box, real cutOff){
      if(forceNextUpdate){
	forceNextUpdate = false;
	return true;
      }
      if(box != currentBox) return true;
      if(cutOff != currentCutOff) return true;
      return false;
    }

  };
}
#endif




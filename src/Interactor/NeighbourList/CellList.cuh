/*Raul P. Pelaez 2020. Cell List UAMMD
  CellList is an instance of the NeighbourList concept.

  A NeighbourList can compute and provide for each particle a list of particle indices that are closer than a certain distance to it.

  There are three ways to interface with NeighbourList:

  1. Providing a Transverser to it
  2. Asking for a NeighbourContainer from it
  3. Any additional mechanism own to the particular instance (a.i. you can get the cell list binning structure with CellList)

  Methods 1. and 2. are the fastest ones it the list is to be used once/twice per construction, as both methods do not force the construction of a neighbour list explicitly.

  See usage for instructions on how to use each method.

  All neighbour list must be capable of:

  -Handling particle groups.
  -Provide an iterator with a list of neighbour particles for each particle*
  -Provide an iterator with the number of neighbours of each particle.
  -Provide a "transverse" method that takes a Transverser and goes through every pair of particles with it (See NBodyForces.cuh or RadialPotential.cuh)
  -Provide a NeighbourContainer able to provide, for each particle, a forward iterator with its neighbours.*

  *Note that this does not imply that a neighbour list is constructed, just that the iterator is able to provide the neighbours.


USAGE:

 //Create a cellList:
 auto cl = make_shared<CellList>(pd, pg, sys);
 //Update a list
 //If you pass a grid, the cellSize should be >= cutoff
 cl->update([A box or a grid], cutOff);

 //THE DIFFERENT INTERFACE METHODS

 //Traverse the list using the internal CellList mechanism(Method 1)
 cl->transverseList([A transverser],cudaStream);

 //Get a NeighbourContainer (Method 2.). This is equivalent to Method 1 (see transverseList kernel), but instead of having to provide a Transverser you can use this structure to get iterators to the neighbours of each particle (as the aforementioned kernel does) and process them manually. This allows for more versatility.
 //See examples/NeighbourListIterator.cu for an example
 auto nc = cl->getNeighbourContainer();

 //Get a cell list to use manually, which provides a spatial binning and a list with the bin of each particle (Method 3.)
 auto cell_list_data = cl->getCellList();

See PairForces.cu or examples/NeighbourListIterator.cu for examples on how to use a NL.

References:

[1] http://developer.download.nvidia.com/assets/cuda/files/particles.pdf
 */
#ifndef CELLLIST_CUH
#define CELLLIST_CUH

#include"System/System.h"
#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"cub/thread/thread_load.cuh"
#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include"utils/cxx_utils.h"
#include"utils/TransverserUtils.cuh"
#include"utils/exception.h"
#include"CellList/CellListBase.cuh"
#include"CellList/NeighbourContainer.cuh"
#include"Interactor/NeighbourList/common.cuh"
#include <limits>
namespace uammd{
  class CellList{
  protected:
    shared_ptr<ParticleData> pd;
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;

    CellListBase cl;
    connection posWriteConnection;

    bool force_next_update = true;
    real3 currentCutOff;
    Box currentBox;

    void handlePosWriteRequested(){
      sys->log<System::DEBUG1>("[CellList] Issuing a list update after positions were written to.");
      force_next_update = true;
    }

    Grid createUpdateGrid(Box box, real3 cutOff){
      real3 L = box.boxSize;
      constexpr real inf = std::numeric_limits<real>::max();
      //If the box is non periodic L and cellDim are free parameters
      //If the box is infinite then periodicity is irrelevan
      constexpr int maximumNumberOfCells = 64;
      if(L.x >= inf) L.x = maximumNumberOfCells*cutOff.x;
      if(L.y >= inf) L.y = maximumNumberOfCells*cutOff.y;
      if(L.z >= inf) L.z = maximumNumberOfCells*cutOff.z;
      Box updateBox(L);
      updateBox.setPeriodicity(box.isPeriodicX() and L.x < inf, box.isPeriodicY() and L.y<inf, box.isPeriodicZ() and L.z<inf);
      Grid a_grid = Grid(updateBox, cutOff);
      int3 cellDim = a_grid.cellDim;
      if(cellDim.x <= 3) cellDim.x = 1;
      if(cellDim.y <= 3) cellDim.y = 1;
      if(cellDim.z <= 3) cellDim.z = 1;
      a_grid = Grid(updateBox, cellDim);
      return a_grid;
    }

  public:

    CellList(shared_ptr<ParticleData> pd, shared_ptr<System> sys = nullptr):
      CellList(pd, std::make_shared<ParticleGroup>(pd), pd->getSystem()){}

    CellList(shared_ptr<ParticleData> pd, shared_ptr<ParticleGroup> pg, shared_ptr<System> sys = nullptr):
      pd(pd), pg(pg), sys(pd->getSystem()), currentCutOff(real3()), currentBox(Box()){
      sys->log<System::MESSAGE>("[CellList] Created");
      posWriteConnection = pd->getPosWriteRequestedSignal()->connect([this](){this->handlePosWriteRequested();});
      CudaCheckError();
    }

    ~CellList(){
      sys->log<System::DEBUG>("[CellList] Destroyed");
      posWriteConnection.disconnect();
    }

    void update(Box box, real cutOff, cudaStream_t st = 0){
      update(box, make_real3(cutOff), st);
    }


    void update(Box box, real3 cutOff, cudaStream_t st = 0){
      if(needsRebuild(box, cutOff)){
	sys->log<System::DEBUG1>("[CellList] Updating cell list");
	currentBox = box;
	currentCutOff = cutOff;
	int numberParticles = pg->getNumberParticles();
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto posGroupIterator = pg->getPropertyIterator(pos);
	Grid grid = createUpdateGrid(box, cutOff);
	cl.update(posGroupIterator, numberParticles, grid, st);
      }
      else{
	sys->log<System::DEBUG1>("[CellList] Ignoring unnecessary update");
      }
    }

    template<class Transverser>
    void transverseList(Transverser &tr, cudaStream_t st = 0){
      sys->log<System::DEBUG2>("[CellList] Transversing Cell List with %s", type_name<Transverser>().c_str());
      const int numberParticles = pg->getNumberParticles();
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      auto globalIndex = pg->getIndexIterator(access::location::gpu);
      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);
      SFINAE::TransverserAdaptor<Transverser>::prepare(tr, pd);
      NeighbourList_ns::transverseWithNeighbourContainer<<<Nblocks, Nthreads, shMemorySize, st>>>(tr,
									       globalIndex,
									       this->getNeighbourContainer(),
									       numberParticles);
      CudaCheckError();
    }

    CellListBase::CellListData getCellList(){
      return cl.getCellList();
    }

    CellList_ns::NeighbourContainer getNeighbourContainer(){
      auto nl = this->getCellList();
      return CellList_ns::NeighbourContainer(nl);
    }

  private:

    bool needsRebuild(Box box, real3 cutOff){
      if(force_next_update) return true;
      if(cutOff.x != currentCutOff.x) return true;
      if(cutOff.y != currentCutOff.y) return true;
      if(cutOff.z != currentCutOff.z) return true;
      if(box != currentBox) return true;
      return false;
    }

  };


}
#endif



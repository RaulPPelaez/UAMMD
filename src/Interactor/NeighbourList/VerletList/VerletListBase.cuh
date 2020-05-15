/*Raul P. Pelaez 2020. Verlet List base implementation.

  A Verlet neighbour list, keeps record of the neighbours of each particle. Each time it is updated with a given cut off distance VerletListBase will compute the neighbours of each particle within a distance a little higher than requested. Subsequent calls to update will only incur the rebuild of this list when a particle has traveled a certain distance related to this extra cut off.

USAGE:
This class does not need any UAMMD structures to work. Can be used as a standalone object.

//Create:
VerletListBase cl;
thrust::device_vector<real4> some_positions(numberParticles);
//fill positions
...
Box someBox(make_real3(32,32,32));
real someCutOff = 2.5;
//Update the neighbour list
nl.update(thrust::raw_pointer_cast(some_positions), numberPartinles, someBox, someCutOff);
//Get a list of neighbours for each particles
auto data = nl.getVerletList();
//set the extra cut off

//Can be coupled with a NeighbourContainer to traverse neighbours or used directly.
//Get a NeighbourContainer
VerletList_ns::NeigbourContainer nc(nl);
//See NeighbourContainer for more info
nl.setCutOffMultiplier(1.1); //10% extra
//Get the number of steps since last update (can be used to optimize the extra cut off).
nl.getNumberOfStepsSinceLastUpdate();

See VerletList.cuh for a UAMMD compatible interface to this utility.
Implementation details:

  VerletListBase uses BaseNeighbourList to construct a verlet list (which in turn uses a cell list). BaseNeighbour List is equivalent to a Verlet list with no extra cut off.

 */
#ifndef VERLETLISTBASE_CUH
#define VERLETLISTBASE_CUH
#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include<thrust/device_vector.h>
#include"Interactor/NeighbourList/BasicList/BasicListBase.cuh"

#include<limits>

namespace uammd{
  namespace VerletListBase_ns{

    template<class PosIterator>
    __global__ void checkMaximumDrift(PosIterator currentPos,
				      real4 *storedPos,
				      real maxDistAllowed,
				      uint* errorFlag,
				      Box box,
				      int numberParticles){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=numberParticles) return;
      real3 currentPos_i = make_real3(currentPos[id]);
      real3 previousPos_i = make_real3(storedPos[id]);
      real3 rij = box.apply_pbc(currentPos_i - previousPos_i);
      if(dot(rij, rij)>=(maxDistAllowed*maxDistAllowed)){
	atomicAdd(errorFlag, 1u);
      }
    }

  }

  class VerletListBase{
  protected:
    BasicNeighbourListBase nl;
    real verletRadiusMultiplier;
    real currentCutOff;
    Box currentBox;
    thrust::device_vector<real4> storedPos;
    thrust::device_vector<uint> errorFlagsGPU;
    thrust::device_vector<real4> sortPos;
    bool forceNextRebuild;
    int stepsSinceLastUpdate = 0;
  public:

    using VerletListData = BasicNeighbourListBase::BasicNeighbourListData;

    VerletListData getVerletList(){
      auto listData = nl.getBasicNeighbourList();
      listData.sortPos = thrust::raw_pointer_cast(sortPos.data());
      return listData;
    }

    VerletListBase(){
      currentCutOff = 0;
      currentBox = Box();
      verletRadiusMultiplier = 1.08;
      errorFlagsGPU.resize(1);
      forceNextRebuild = true;
    }

    ~VerletListBase(){}

    template<class PositionIterator>
    void update(PositionIterator pos, int numberParticles, Box box, real cutOff, cudaStream_t st = 0){
      if(needsRebuild(pos, numberParticles, box, cutOff, st)){
	stepsSinceLastUpdate = 0;
	currentBox = box;
	currentCutOff = cutOff;
	storeCurrentPos(pos, numberParticles, st);
	rebuildList(st);
      }
      updateSortedPositions(pos, numberParticles, st);
      stepsSinceLastUpdate++;
    }

    void setCutOffMultiplier(real newMultiplier){
      forceNextRebuild = true;
      this->verletRadiusMultiplier = newMultiplier;
    }

    int getNumberOfStepsSinceLastUpdate(){
      return stepsSinceLastUpdate-1;
    }

    void forceNextUpdate(){
      this->forceNextRebuild = true;
    }

  private:

    template<class PositionIterator>
    void storeCurrentPos(PositionIterator pos, int numberParticles, cudaStream_t st = 0){
      storedPos.resize(numberParticles);
      thrust::copy(thrust::cuda::par.on(st), pos, pos + numberParticles, storedPos.begin());
      CudaCheckError();
    }

    template<class PositionIterator>
    void updateSortedPositions(PositionIterator pos, int numberParticles, cudaStream_t st = 0){
      sortPos.resize(numberParticles);
      auto listData = nl.getBasicNeighbourList();
      auto reorderIt = thrust::make_permutation_iterator(pos, listData.groupIndex);
      thrust::copy(thrust::cuda::par.on(st), reorderIt, reorderIt + numberParticles, sortPos.begin());
    }

    void rebuildList(cudaStream_t st){
      real rcut = currentCutOff*verletRadiusMultiplier;
      nl.update(storedPos.begin(), storedPos.size(), currentBox, rcut, st);
    }

    template<class PositionIterator>
    bool needsRebuild(PositionIterator pos, int numberParticles, Box box, real cutOff, cudaStream_t st = 0){
      if(forceNextRebuild){
	forceNextRebuild = false;
        return true;
      }
      if(box != currentBox or cutOff != currentCutOff or numberParticles != storedPos.size()){
	return true;
      }
      if(isParticleDriftOverThreshold(pos, numberParticles, st)){
	System::log<System::DEBUG2>("[VerletList] Particle drift forced rebuild. Last one %d steps ago.",
				     stepsSinceLastUpdate);
	return true;
      }
      return false;
    }

    template<class PositionIterator>
    bool isParticleDriftOverThreshold(PositionIterator pos, int numberParticles, cudaStream_t st = 0){
      real thresholdDistance = (verletRadiusMultiplier*currentCutOff-currentCutOff)/2.0;
      errorFlagsGPU[0] = 0;
      int blockSize = 128;
      int nblocks = numberParticles/blockSize+1;
      VerletListBase_ns::checkMaximumDrift<<<nblocks, blockSize, 0, st>>>(pos,
								  thrust::raw_pointer_cast(storedPos.data()),
								  thresholdDistance,
								  thrust::raw_pointer_cast(errorFlagsGPU.data()),
								  currentBox,
								  numberParticles);
      CudaCheckError();
      uint errorFlag = errorFlagsGPU[0];
      if(errorFlag>0){
	System::log<System::DEBUG2>("[VerletList] Found %d particles over threshold.", errorFlag);
      }
      bool isSomeParticleDisplacementOverThreshold = errorFlag>0;
      return isSomeParticleDisplacementOverThreshold;
    }

  };

}
#endif

/*Raul P. Pelaez 2020. Basic Neighbour List

USAGE:
This class does not need any UAMMD structures to work. Can be used as a
standalone object.

//Create:
BasicNeighbourListBase cl;
thrust::device_vector<real4> some_positions(numberParticles);
//fill positions
...
Box someBox(make_real3(32,32,32));
real someCutOff = 2.5;
//Update the neighbour list
cl.update(thrust::raw_pointer_cast(some_positions), numberParticles, someBox,
someCutOff);
//Get a list of neighbours for each particles
auto data = cl.getBasicNeighbourList();

//Can be coupled with a NeighbourContainer to traverse neighbours or used
directly.
//Get a NeighbourContainer
BasicNeighbourList_ns::NeigbourContainer nc(cl);
//See NeighbourContainer for more info

Implementation notes:
This class maintains a CellList and uses it to generate a neighbour list. It
should be expected to be better than using a CellList directly if the neighbour
list is to be used several times between updates.
*/
#ifndef BASICNEIGHBOURLISTBASE_CUH
#define BASICNEIGHBOURLISTBASE_CUH

#include "Interactor/NeighbourList/CellList/CellListBase.cuh"
#include "Interactor/NeighbourList/CellList/NeighbourContainer.cuh"
#include "utils/Box.cuh"
#include "utils/Grid.cuh"
#include <thrust/device_vector.h>

namespace uammd {
namespace BasicNeighbourList_ns {
__global__ void fillBasicNeighbourList(CellList_ns::NeighbourContainer ni,
                                       int *neighbourList,
                                       int *numberNeighbours,
                                       int maxNeighboursPerParticle,
                                       real cutOff2, int numberParticles,
                                       Box box, int *tooManyNeighboursFlag) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= numberParticles)
    return;
  int nneigh = 0;
  // const int offset = id*maxNeighboursPerParticle;
  const real3 pi =
      make_real3(cub::ThreadLoad<cub::LOAD_LDG>(ni.getSortedPositions() + id));
  ni.set(id);
  auto it = ni.begin();
  while (it) {
    auto n = *it++;
    const int cur_j = n.getInternalIndex();
    const real3 pj = make_real3(
        cub::ThreadLoad<cub::LOAD_LDG>(ni.getSortedPositions() + cur_j));
    const real3 rij = box.apply_pbc(pj - pi);
    if (dot(rij, rij) <= cutOff2) {
      nneigh++;
      if (nneigh >= maxNeighboursPerParticle) {
        atomicMax(tooManyNeighboursFlag, nneigh);
        return;
      }
      neighbourList[(nneigh - 1) * numberParticles + id] = cur_j;
    }
  }
  numberNeighbours[id] = nneigh;
}
} // namespace BasicNeighbourList_ns

class BasicNeighbourListBase {
protected:
  CellListBase cl;
  real currentCutOff;
  Box currentBox;
  thrust::device_vector<int> neighbourList, numberNeighbours;
  thrust::device_vector<int> errorFlags;

  int maxNeighboursPerParticle;

  struct NeighbourListOffsetFunctor {
    NeighbourListOffsetFunctor(int str) : stride(str) {}
    int stride;
    inline __host__ __device__ int operator()(const int &index) const {
      // return index*stride;
      return stride;
    }
  };

  using CountingIterator = thrust::counting_iterator<int>;
  using StrideIterator = thrust::transform_iterator<NeighbourListOffsetFunctor,
                                                    CountingIterator, int>;

  Grid createUpdateGrid(Box box, real cutOff) {
    real3 L = box.boxSize;
    constexpr real inf = std::numeric_limits<real>::max();
    // If the box is non periodic L and cellDim are free parameters
    // If the box is infinite then periodicity is irrelevan
    constexpr int maximumNumberOfCells = 64;
    if (L.x >= inf)
      L.x = maximumNumberOfCells * cutOff;
    if (L.y >= inf)
      L.y = maximumNumberOfCells * cutOff;
    if (L.z >= inf)
      L.z = maximumNumberOfCells * cutOff;
    Box updateBox(L);
    updateBox.setPeriodicity(box.isPeriodicX() and L.x < inf,
                             box.isPeriodicY() and L.y < inf,
                             box.isPeriodicZ() and L.z < inf);
    Grid a_grid = Grid(updateBox, cutOff);
    int3 cellDim = a_grid.cellDim;
    if (cellDim.x <= 3)
      cellDim.x = 1;
    if (cellDim.y <= 3)
      cellDim.y = 1;
    if (cellDim.z <= 3)
      cellDim.z = 1;
    a_grid = Grid(updateBox, cellDim);
    return a_grid;
  }

public:
  BasicNeighbourListBase() {
    maxNeighboursPerParticle = 32;
    errorFlags.resize(1);
  }

  ~BasicNeighbourListBase() {}

  template <class PositionIterator>
  void update(PositionIterator pos, int numberParticles, Box box, real cutOff,
              cudaStream_t st = 0) {
    currentBox = box;
    currentCutOff = cutOff;
    Grid grid = createUpdateGrid(box, cutOff);
    cl.update(pos, numberParticles, grid, st);
    resizeNeighbourListToCurrent(numberParticles);
    fillBasicNeighbourList(st);
    CudaCheckError();
  }

  struct BasicNeighbourListData {
    const int *neighbourList;
    const int *numberNeighbours;
    const real4 *sortPos;
    const int *groupIndex;
    StrideIterator particleStride =
        StrideIterator(CountingIterator(0), NeighbourListOffsetFunctor(0));
  };

  BasicNeighbourListData getBasicNeighbourList(cudaStream_t st = 0) {
    BasicNeighbourListData nl;
    auto cld = cl.getCellList();
    nl.neighbourList = thrust::raw_pointer_cast(neighbourList.data());
    nl.numberNeighbours = thrust::raw_pointer_cast(numberNeighbours.data());
    nl.sortPos = cld.sortPos;
    nl.groupIndex = cld.groupIndex;
    // nl.particleStride = StrideIterator(CountingIterator(0),
    // NeighbourListOffsetFunctor(maxNeighboursPerParticle));
    nl.particleStride =
        StrideIterator(CountingIterator(0),
                       NeighbourListOffsetFunctor(numberNeighbours.size()));
    return nl;
  }

private:
  void resizeNeighbourListToCurrent(int numberParticles) {
    neighbourList.resize(numberParticles * (maxNeighboursPerParticle + 1));
    numberNeighbours.resize(numberParticles);
  }

  void fillBasicNeighbourList(cudaStream_t st) {
    while (not tryToFillNeighbourList(st)) {
      increaseMaximumNeighboursPerParticle();
    }
  }

  bool tryToFillNeighbourList(cudaStream_t st) {
    System::log<System::DEBUG3>(
        "[BasicList] Attempting to fill list with %d neighbours per particle",
        maxNeighboursPerParticle);
    int numberParticles = numberNeighbours.size();
    auto cldata = cl.getCellList();
    auto ni = CellList_ns::NeighbourContainer(cldata);
    auto neighbourList_ptr = thrust::raw_pointer_cast(neighbourList.data());
    auto numberNeighbours_ptr =
        thrust::raw_pointer_cast(numberNeighbours.data());
    errorFlags[0] = 0;
    int *d_tooManyNeighboursFlag = thrust::raw_pointer_cast(errorFlags.data());
    int Nthreads = 128;
    int Nblocks =
        numberParticles / Nthreads + ((numberParticles % Nthreads) ? 1 : 0);
    BasicNeighbourList_ns::fillBasicNeighbourList<<<Nblocks, Nthreads, 0, st>>>(
        ni, neighbourList_ptr, numberNeighbours_ptr, maxNeighboursPerParticle,
        currentCutOff * currentCutOff, numberParticles, currentBox,
        d_tooManyNeighboursFlag);
    CudaCheckError();
    int flag = errorFlags[0];
    bool foundTooManyNeighbours = flag != 0;
    return not foundTooManyNeighbours;
  }

  void increaseMaximumNeighboursPerParticle() {
    this->maxNeighboursPerParticle += 32;
    System::log<System::DEBUG3>(
        "[BasicList] Increasing maximum number of neighbours to %d",
        maxNeighboursPerParticle);
    resizeNeighbourListToCurrent(numberNeighbours.size());
  }
};

} // namespace uammd
#endif

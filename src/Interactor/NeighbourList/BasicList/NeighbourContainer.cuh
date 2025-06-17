/* Raul P. Pelaez 2020. Neighbour Container for Basic Neighbour List.

TODO:
100- Profile NeighbourContainer::increment()
 */
#ifndef BASICNEIGHBOURLISTNEIGHBOURCONTAINER_CUH
#define BASICNEIGHBOURLISTNEIGHBOURCONTAINER_CUH
#include "Interactor/NeighbourList/BasicList/BasicListBase.cuh"
namespace uammd {

namespace BasicNeighbourList_ns {
class NeighbourContainer; // forward declaration for befriending

class NeighbourIterator; // forward declaration for befriending

struct Neighbour {
  __device__ Neighbour(const Neighbour &other)
      : internal_i(other.internal_i), groupIndex(other.groupIndex),
        sortPos(other.sortPos) {}

  __device__ int getInternalIndex() { return internal_i; }

  __device__ int getGroupIndex() { return groupIndex[internal_i]; }

  __device__ real4 getPos() {
    return cub::ThreadLoad<cub::LOAD_LDG>(sortPos + internal_i);
  }

private:
  int internal_i;
  const int *groupIndex;
  const real4 *sortPos;
  friend class NeighbourIterator;

  Neighbour() = delete;

  __device__ Neighbour(int i, const int *gi, const real4 *sp)
      : internal_i(i), groupIndex(gi), sortPos(sp) {}
};

// This forward iterator must be constructed by NeighbourContainer,
class NeighbourIterator
    : public thrust::iterator_adaptor<
          NeighbourIterator, int, Neighbour, thrust::any_system_tag,
          thrust::forward_device_iterator_tag, Neighbour, int> {
  friend class thrust::iterator_core_access;
  friend class NeighbourContainer;
  int particleIndex;
  int currentNeighbourIndex;
  int currentNeighbourCounter;
  BasicNeighbourListBase::BasicNeighbourListData nl;
  int numberNeighbours;
  int firstNeighbourIndex;
  static constexpr int noMoreNeighbours = -1;
  // Go to the next neighbour
  __device__ void increment() {
    currentNeighbourCounter++;
    currentNeighbourCounter = currentNeighbourCounter >= numberNeighbours
                                  ? noMoreNeighbours
                                  : currentNeighbourCounter;
    int counter = currentNeighbourCounter;
    // int index =
    // counter!=noMoreNeighbours?nl.neighbourList[firstNeighbourIndex +
    // counter]:noMoreNeighbours;
    int index =
        counter != noMoreNeighbours
            ? nl.neighbourList[counter * firstNeighbourIndex + particleIndex]
            : noMoreNeighbours;
    currentNeighbourIndex = index;
  }

  __device__ Neighbour dereference() const {
    return Neighbour(currentNeighbourIndex, nl.groupIndex, nl.sortPos);
  }

  // Can only be advanced
  __device__ void decrement() = delete;
  // Just a forward iterator
  __device__ Neighbour operator[](int i) = delete;

  __device__ bool equal(NeighbourIterator const &other) const {
    return other.particleIndex == particleIndex and
           other.currentNeighbourCounter == currentNeighbourCounter;
  }

  __device__
  NeighbourIterator(int i, BasicNeighbourListBase::BasicNeighbourListData nl,
                    bool begin)
      : particleIndex(i), currentNeighbourCounter(-1),
        currentNeighbourIndex(noMoreNeighbours), nl(nl) {
    if (begin) {
      numberNeighbours = nl.numberNeighbours[i];
      firstNeighbourIndex = nl.particleStride[i];
      increment();
    } else {
      currentNeighbourCounter = noMoreNeighbours;
    }
  }

public:
  __device__ operator bool() {
    return currentNeighbourIndex != noMoreNeighbours;
  }
};

struct NeighbourContainer {
  int my_i = -1;
  BasicNeighbourListBase::BasicNeighbourListData nl;
  NeighbourContainer(BasicNeighbourListBase::BasicNeighbourListData nl)
      : nl(nl) {}

  __device__ void set(int i) { this->my_i = i; }

  __device__ NeighbourIterator begin() {
    return NeighbourIterator(my_i, nl, true);
  }

  __device__ NeighbourIterator end() {
    return NeighbourIterator(my_i, nl, false);
  }

  __host__ __device__ const real4 *getSortedPositions() { return nl.sortPos; }

  __host__ __device__ const int *getGroupIndexes() { return nl.groupIndex; }
};
} // namespace BasicNeighbourList_ns
} // namespace uammd

#endif

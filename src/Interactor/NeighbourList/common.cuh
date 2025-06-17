/*Raul P. Pelaez. 2020. Common utilities for Neighbour lists
 */
#ifndef NEIGHBOURLIST_COMMON_CUH
#define NEIGHBOURLIST_COMMON_CUH
#include "third_party/uammd_cub.cuh"
#include "utils/TransverserUtils.cuh"
namespace uammd {
namespace NeighbourList_ns {

template <class Transverser, class NeighbourContainer, class IndexIterator>
__global__ void transverseWithNeighbourContainer(Transverser tr,
                                                 IndexIterator globalIndex,
                                                 NeighbourContainer ni, int N) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N)
    return;
  const int ori = globalIndex[ni.getGroupIndexes()[id]];
  const real4 pi = cub::ThreadLoad<cub::LOAD_LDG>(ni.getSortedPositions() + id);
  using Adaptor = SFINAE::TransverserAdaptor<Transverser>;
  Adaptor adaptor;
  auto quantity = Adaptor::zero(tr);
  adaptor.getInfo(tr, ori);
  // for(auto n: ni){
  ni.set(id);
  auto it = ni.begin();
  while (it) {
    auto neighbour = *it++;
    const real4 pj = neighbour.getPos();
    const int global_index = globalIndex[neighbour.getGroupIndex()];
    Adaptor::accumulate(tr, quantity,
                        adaptor.compute(tr, global_index, pi, pj));
  }
  tr.set(ori, quantity);
}

} // namespace NeighbourList_ns
} // namespace uammd

#endif

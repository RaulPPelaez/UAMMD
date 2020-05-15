/*Raul P. Pelaez. 2020. Common utilities for Neighbour lists
 */
#ifndef NEIGHBOURLIST_COMMON_CUH
#define NEIGHBOURLIST_COMMON_CUH
#include"cub/thread/thread_load.cuh"
#include"utils/TransverserUtils.cuh"
namespace uammd{
  namespace NeighbourList_ns{

    template<class Transverser, class NeighbourContainer, class IndexIterator>
    __global__ void transverseWithNeighbourContainer(Transverser tr,
						     IndexIterator globalIndex,
						     NeighbourContainer ni,
						     int N){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id >= N) return;
      const int ori = globalIndex[ni.getGroupIndexes()[id]];
#if CUB_PTX_ARCH < 300
      constexpr auto cubModifier = cub::LOAD_DEFAULT;
#else
      constexpr auto cubModifier = cub::LOAD_LDG;
#endif
      const real4 pi = cub::ThreadLoad<cubModifier>(ni.getSortedPositions() + id);
      auto quantity = tr.zero();
      SFINAE::Delegator<Transverser> del;
      del.getInfo(tr, ori);
      //for(auto n: ni){
      ni.set(id);
      auto it = ni.begin();
      while(it){
	auto neighbour = *it++;
	const real4 pj = neighbour.getPos();
	const int global_index = globalIndex[neighbour.getGroupIndex()];
	tr.accumulate(quantity, del.compute(tr, global_index, pi, pj));
      }
      tr.set(ori, quantity);
    }

  }
}

#endif

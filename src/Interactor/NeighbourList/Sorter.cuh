#ifndef SORTER_CUH
#define SORTER_CUH

#include"utils/utils.h"
#include"globals/defines.h"
//#include"NeighbourList_common.cuh"
namespace Sorter{

  /*Radix sort by key using cub, puts sorted versions of inde,hash in index_alt, hash_alt*/
  void sortByKey(uint *&index, uint *&index_alt, uint *&hash, uint *&hash_alt, int N);
    /*Reorder arrays with the new order, and transform if needed*/
  template<class T>
  __global__ void reorderProperty(T* __restrict__ old,
				  T* __restrict__  sorted,
				  uint* __restrict__ pindex, int N){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;    
    sorted[i] = old[pindex[i]];
  }

  /*In case old position is a texture*/
  template<class T>
  __global__ void reorderProperty(TexReference old,
				  T  * __restrict__ sorted,
				  uint* __restrict__ pindex, int N){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;
    sorted[i] = tex1Dfetch<T>(old, pindex[i]);
  }

  template<class Told, class T, class TransformFunction>
  __global__ void reorderTransformProperty(TexReference old,
					   T* __restrict__  sorted,
					   uint* __restrict__ pindex,
					   TransformFunction tf, int N){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;   
    sorted[i] = tf(tex1Dfetch<Told>(old, pindex[i]));
  }
  template<class Told, class T, class TransformFunction>
  __global__ void reorderTransformProperty(Told* __restrict__ old,
					   T* __restrict__  sorted,
					   uint* __restrict__ pindex,
					   TransformFunction tf, int N){
    uint i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;   
    sorted[i] = tf(old[pindex[i]]);
  }
  
}

#endif
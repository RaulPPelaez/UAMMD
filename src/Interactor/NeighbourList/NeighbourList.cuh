/*Raul P.Pelaez 2017. Neighbour List base class definition

  A neighbour list can be updated (makeNeighbourList() ) and transversed (NeighbourList::transverseList[General](NL nl, TR tr)). Additionally, it can sort the particles to increase data locality when transversing the list.


  See https://github.com/RaulPPelaez/UAMMD/wiki/Neighbour-List for more info

*/
#ifndef NEIGHBOURLIST_H
#define NEIGHBOURLIST_H


#include"Sorter.cuh"
#include"globals/globals.h"
#include"globals/defines.h"
#include"NeighbourList_common.cuh"
class NeighbourList_Base{
public:
  NeighbourList_Base():NeighbourList_Base(gcnf.rcut, gcnf.L, gcnf.N, true){}
  NeighbourList_Base(real rcut, real3 L, int N, bool reorder = true);
  virtual ~NeighbourList_Base(){ }
  void reorderParticles(cudaStream_t st = 0);
  
  /*Reorder some array to the new order, perfom some transformation if needed*/
  template<class Told, class T, class TransformFunction>
  void reorderTransformProperty(Told* old, T* sorted, TransformFunction ft, int N, cudaStream_t st=0);
  template<class Told, class T, class TransformFunction>
  void reorderTransformProperty(TexReference old, T* sorted, TransformFunction ft, int N, cudaStream_t st=0);
  
  template<class T>
  void reorderProperty(T* old, T* sorted, int N, cudaStream_t st=0);
  template<class T>
  void reorderProperty(TexReference old, T* sorted, int N, cudaStream_t st=0);

  virtual void makeNeighbourList(cudaStream_t st = 0) = 0;  

  virtual bool needsReorder() = 0;
  /*TransverseList is needed in each implementations .cuh
    It must be a tempalted function, see cellList and PairForces.
    I do not know how to declare this function inside the class.*/
  /*Each implementation must specialize the template transverseList, see below*/
  real getRcut(){ return this->rcut;}
  uint *getOriginalOrder(){ return particleIndex.d_m;}
  virtual void print(){}
protected:
  int N;
  real rcut;
  real3 L;

  uint last_step_updated; //Last step in which the list was updated

  GPUVector<real4> sortPos;
  GPUVector<uint> particleHash, particleIndex;
  GPUVector<uint> particleHash_alt, particleIndex_alt;
  NeighbourList::Utils utils;
  int BLOCKSIZE;
};

namespace NeighbourList{
/*Implement this function for any new neighbour list!! do it in the .h*/
  template<class NL, class Transverser>
  __global__ void transverseList(NL nl, Transverser tr);
  template<class NL, class Transverser>
  __global__ void transverseListGeneral(NL nl, Transverser tr);

  /*Call with NeighbourList::transverseList(nl, tr); */
  /*Being nl an instance of a struct containing the needed information inside the new NL and tr an user defined struct*/
}


/*Specializations of reorderProperty and reorderTransformPorperty*/
template<class Told, class T, class TransformFunction>
void NeighbourList_Base::reorderTransformProperty(Told* old, T* sorted, TransformFunction ft, int N, cudaStream_t st){
  if(!this->needsReorder()) return;
  int nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int nblocks  =  N/nthreads +  ((N%nthreads!=0)?1:0);     
  Sorter::reorderTransformProperty<Told,T, TransformFunction><<<nblocks, nthreads,0,st>>>(old, sorted, particleIndex.d_m, ft, N);
}
template<class Told, class T, class TransformFunction>
void NeighbourList_Base::reorderTransformProperty(TexReference old, T* sorted, TransformFunction ft, int N, cudaStream_t st){
  if(!this->needsReorder()) return;
  int nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int nblocks  =  N/nthreads +  ((N%nthreads!=0)?1:0);     
  Sorter::reorderTransformProperty<Told,T, TransformFunction><<<nblocks, nthreads, 0, st>>>(old, sorted, particleIndex.d_m, ft, N);
}

template<class T>
void NeighbourList_Base::reorderProperty(T* old, T* sorted, int N, cudaStream_t st){
  if(!this->needsReorder()) return;
  int nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int nblocks  =  N/nthreads +  ((N%nthreads!=0)?1:0);     
  Sorter::reorderProperty<T><<<nblocks, nthreads, 0, st>>>(old, sorted, particleIndex.d_m, N);
}
    
/*Told can be accesed as a device pointer of type T or a TexReference*/
template<class T>
void NeighbourList_Base::reorderProperty(TexReference old, T* sorted, int N, cudaStream_t st){
  if(!this->needsReorder()) return;
  int nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int nblocks  =  N/nthreads +  ((N%nthreads!=0)?1:0); 
  Sorter::reorderProperty<T><<<nblocks, nthreads, 0, st>>>(old, sorted, particleIndex.d_m, N);
}



#endif

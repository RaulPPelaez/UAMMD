#ifndef VERLETLIST_CUH
#define VERLETLIST_CUH
#include"NeighbourList.cuh"
#include"globals/globals.h"
#include"globals/defines.h"
#include"utils/SFINAE.cuh"
#include"CellList.cuh"

class VerletList: public NeighbourList_Base{
public:
  VerletList(){}
  VerletList(real rcut, real3 L, int N);
  ~VerletList(){ }
  /*Use reoder to construct the list*/
  bool needsReorder(){ return false;}

  /*Override virtual methods*/
  void makeNeighbourList() override;
 
  /*To transverse the neighbours I need the base parameters
    plus texture references to the cell list*/
  struct TransverseInfo{    
    cudaTextureObject_t texPos;
    int* nlist, nneigh;
    int maxNeigh;
    int N;
  };
  /*Although not a virtual method, this function must exist and be defined exactly like this*/
  /*Defined below, see transverseList below to see how to implement a transverser*/
  template<class Transverser>
  void transverse(Transverser &tr)/*override*/;

  void print() override{
    std::cerr<<"\t\tCut-off distance: "<<rcut<<std::endl;

  }
  
private:
  CellList nl;
  
  TransverseInfo nl;
};

namespace NeighbourList{  
  template<class Transverser>
  __global__ void transverseList(VerletList::TransverseInfo nl, Transverser T){
    int ii =  blockIdx.x*blockDim.x + threadIdx.x;

    //Grid-stride loop
    for(int index = ii; index<nl.N; index += blockDim.x * gridDim.x){
      /*Compute force acting on particle particleIndex[index], index in the new order*/
      /*Get my particle's.data*/

      real4 pi = tex1Dfetch<real4>(nl.texPos, index);
      /*Delegator makes possible to call transverseList with a simple Transverser
	(a transverser that only uses positions, with no getInfo method) by using
	SFINAE*/
      /*Note that in the case of a simple Transverser, this code gets trashed by the compiler
	and no additional cost occurs. And with inlining, even when T is general no overhead
        comes from Delegator
      */
      SFINAE::Delegator<Transverser> del;      
      del.getInfo(T, index);

      /*Initial value of the quantity,
	mostly just a hack so the compiler can guess the type of the quantity*/
      auto quantity = T.zero();
      
      int head = nl.maxNeigh*index;
      int nneigh = nl.nneigh[index];

      for(int j = 0; j< nneigh; j++){
	/*Retrieve j info*/
	real4 pj = tex1Dfetch<real4>(nl.texPos, nl.nlist[head+j]);
	/*Ask Delegator for any additional info,
	  compute interaction ij and accumulate the result.*/
	T.accumulate(quantity, del.compute(T, head+j, pi, pj));
       
      }
      	    
      T.set(index, quantity);
    }
    
  }

}

template<class Transverser>
void VerletList::transverse(Transverser &tr){
  int nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int nblocks  =  N/nthreads +  ((N%nthreads!=0)?1:0);
  
  NeighbourList::transverseList<<<nblocks, nthreads>>>(nl, tr);  
}

#endif
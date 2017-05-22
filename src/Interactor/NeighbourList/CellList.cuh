/*Raul P. Pelaez 2017. Cell List headers and template definitions.

  Cell List inherits from NeighbourList_Base.

  Uses the NeighbourList_Base sorting functionality to create a cell list.

  Base sorts the particles by cell index. CellList takes this sorted positions and fills to arrays, cellStart and cellEnd, with the indices in this array where each cell starts and ends.

  To go through the neighbours of a given particle, cellList goes through all the particles in the given particle's cell and its 27 neighbours.

So, apart from the base class. CellList only needs these two arrays of size ncells.


 */
#ifndef CELLLIST_CUH
#define CELLLIST_CUH
#include"NeighbourList.cuh"
#include"globals/globals.h"
#include"globals/defines.h"
#include"utils/SFINAE.cuh"


/*NeighbourList_Base orders the particles according to their cell index.
  CellList just takes this sorted array of positions and finds where each
  cell starts and ends.
  This way one has fast access to all particles in a given cell just by 
  looping from sortPos[cellStart[icell]] to sortPos[cellEnd[icell]]
*/

class Mesh{
public:  
  NeighbourList::Utils utils;
  Mesh(){}
  Mesh(real rcut, real3 L):
    rcut(rcut), L(L){
    
    cellDim = make_int3(L/rcut);
    if(gcnf.D2) cellDim.z = 1;
    
    real3 cellSize = L/make_real3(cellDim);

    utils.cellDim = cellDim;
    ncells = cellDim.x*cellDim.y*cellDim.z;
    utils.cellSize = cellSize;
    utils.invCellSize = 1.0/cellSize;
    if(gcnf.D2) utils.invCellSize.z = real(0.0);
    utils.gridPos2CellIndex = make_int3( 1,
					 utils.cellDim.x,
					 utils.cellDim.x*utils.cellDim.y);
    utils.Lhalf = L*0.5;
  }
  Mesh(int3 cellDim, real3 L, real rcut): cellDim(cellDim), L(L), rcut(rcut){
    
    real3 cellSize = L/make_real3(cellDim);
    utils.cellSize = cellSize;
    utils.cellDim = cellDim;

    ncells = cellDim.x*cellDim.y*cellDim.z;
    utils.invCellSize = 1.0/cellSize;
    if(gcnf.D2) utils.invCellSize.z = real(0.0);
    utils.gridPos2CellIndex = make_int3( 1,
					 utils.cellDim.x,
					 utils.cellDim.x*utils.cellDim.y);
    utils.Lhalf = L*0.5;    
  }

  void print(){
    std::cerr<<"\t\tCut-off distance: "<<rcut<<std::endl;
    std::cerr<<"\t\tNumber of cells: "<<utils.cellDim.x<<" "<<utils.cellDim.y<<" "<<utils.cellDim.z;
    std::cerr<<"; Total cells: "<<utils.cellDim.x*utils.cellDim.y*utils.cellDim.z<<std::endl;
  }

  real rcut;
  real3 L;
  int3 cellDim;
  int ncells;
};

class CellList: public NeighbourList_Base{
public:
  CellList():CellList(gcnf.rcut, gcnf.L, gcnf.N){}
  CellList(real rcut):CellList(rcut, gcnf.L, gcnf.N){}
  CellList(real rcut, real3 L, int N);
  ~CellList(){ }
  /*Use reoder to construct the list*/
  bool needsReorder(){ return true;}

  /*Override virtual methods*/
  void makeNeighbourList(cudaStream_t st=0) override;
 
  /*To transverse the neighbours I need the base parameters
    plus texture references to the cell list*/
  struct TransverseInfo{
    NeighbourList::Utils utils; /*Contains device functions and parameters such as getCell(pos)*/
    TexReference texCellStart, texCellEnd;
    TexReference texSortPos; /*Texture reference to sorted positions*/
    uint *particleIndex;     /*Indices of particles in the original array*/
    int N;
  };
  /*Although not a virtual method, this function must exist and be defined exactly like this*/
  /*Defined below, see transverseList below to see how to implement a transverser*/
  template<class Transverser>
  void transverse(Transverser &tr, cudaStream_t st=0)/*override*/;

  void print() override{
    std::cerr<<"\t\tCut-off distance: "<<rcut<<std::endl;
    std::cerr<<"\t\tNumber of cells: "<<utils.cellDim.x<<" "<<utils.cellDim.y<<" "<<utils.cellDim.z;
    std::cerr<<"; Total cells: "<<utils.cellDim.x*utils.cellDim.y*utils.cellDim.z<<std::endl;

  }
  
private:
  /*The only additional things I need are this two arrays containing where
    particles in each cell start and end in sortPos*/
  GPUVector<uint> cellStart, cellEnd;
  TransverseInfo nl;

};

namespace NeighbourList{
  /*Transverse the list and use the methods in transverser to make
    each neighbour pair interact. You can see examples in Pairforces.cu and PairForcesDPD.cu*/
  /*In the simpler case, Transverser must be a struct
     that implement the following methods:
       1.-Initial value of returnInfo (i.e real4 and make_real4(0))
           inline __device__ returnInfo zero()
       2.-Compute interaction between a neighbour pair, on i. 
           inline __device__ returnInfo compute(real4 posi, real4 posj);	   
       3.-How to stack results from each neighbour 
          overload += operator of returnInfo, i.e do nothing in case of i.e real4
       4.-Take care of final results (i.e. write to global memory)
          inline __device__ void set(int pi, returnInfo total)    
   */
  /* A more general Transverser can also implement:
       - A method called particleInfo getInfo(int id).
                       That returns any information about particle id(id in sorted order)     
       - A compute method with 4 arguments instead of two:
                returnInfo compute(real4 posi, real4 posj, particleInfo infoi, particleInfo infoj)
                That computes returnInfo with the additional data from getInfo
     Both types of transversers can be use indistinctly by tramnsverseList. So either your transverser is simple or general, just call nl.transverseList.

     You can see examples of transversers in PairForces.cu and PairForcesDPD.cu
     See https://github.com/RaulPPelaez/UAMMD/wiki/List-Transverser for more information
  */
  
  template<class Transverser>
  __global__ void transverseList(CellList::TransverseInfo nl, Transverser T){
    int ii =  blockIdx.x*blockDim.x + threadIdx.x;

    //Grid-stride loop
    for(int index = ii; index<nl.N; index += blockDim.x * gridDim.x){
      /*Compute force acting on particle particleIndex[index], index in the new order*/
      /*Get my particle's.data*/

      real4 pi = tex1Dfetch<real4>(nl.texSortPos, index);
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
      
      int3 celli = nl.utils.getCell(make_real3(pi));
    
      int x,y,z;
      int3 cellj;
      /**Go through all neighbour cells**/
      //For some reason unroll doesnt help here
      int zi = -1; //For the 2D case
      int zf = 1;
      if(gcnfGPU.D2){
	zi = zf = 0;
      }      
      for(x=-1; x<=1; x++)
	for(z=zi; z<=zf; z++)
	  for(y=-1; y<=1; y++){
	    cellj = celli + make_int3(x,y,z);
	    nl.utils.pbc_cell(cellj);
	      
	    uint icell  = nl.utils.getCellIndex(cellj);
	    /*Index of the first particle in the cell's list*/
	    uint firstParticle = tex1Dfetch<uint>(nl.texCellStart, icell);
	    if(firstParticle ==0xffFFffFF) continue; /*Continue only if there are particles in this cell*/
	    /*Index of the last particle in the cell's list*/
	    uint lastParticle = tex1Dfetch<uint>(nl.texCellEnd, icell);
	    uint nincell = lastParticle-firstParticle;
	    /*Because the list is ordered, all the particle indices in the cell are coalescent!*/
	    for(uint j=0; j<nincell; j++){
	      int cur_j = j+firstParticle;
	      /*Retrieve j info*/
	      real4 pj = tex1Dfetch<real4>(nl.texSortPos, cur_j);	      	      
	      /*Ask Delegator for any additional info,
		compute interaction ij and accumulate the result.*/
	      T.accumulate(quantity, del.compute(T, cur_j, pi, pj));
	    }
	    
	  }
      /*Write quantity with the original order to global memory*/
#if __CUDA_ARCH__>=350
      uint oi = __ldg(nl.particleIndex+index);
#else
      uint oi = nl.particleIndex[index]; 
#endif
      T.set(oi, quantity);
    }
    
  }

}

template<class Transverser>
void CellList::transverse(Transverser &tr, cudaStream_t st){
  int nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int nblocks  =  N/nthreads +  ((N%nthreads!=0)?1:0);
  nl.texCellStart = cellStart.getTexture();
  nl.texCellEnd = cellEnd.getTexture();
  nl.texSortPos = sortPos.getTexture();
  nl.particleIndex = particleIndex.d_m;
  
  NeighbourList::transverseList<<<nblocks, nthreads, 0, st>>>(nl, tr);  
}

#endif
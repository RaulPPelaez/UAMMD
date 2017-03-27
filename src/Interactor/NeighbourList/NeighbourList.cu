/*Raul P. Pelaez 2017. NeighbourList_Base virtual base class implementation. 
Takes care of ordering the particles if needed

See https://github.com/RaulPPelaez/UAMMD/wiki/Neighbour-List for more info
*/
#include"NeighbourList.cuh"




NeighbourList_Base::NeighbourList_Base(real rcut, real3 L, int N, bool reorder):
  N(N),
  rcut(rcut),
  L(L),
  last_step_updated(0xffFFff),
  BLOCKSIZE(128){
  if(reorder){
    sortPos =           GPUVector4(N);
    particleIndex =     GPUVector<uint>(N);
    particleHash  =     GPUVector<uint>(N);
    particleIndex_alt = GPUVector<uint>(N);
    particleHash_alt  = GPUVector<uint>(N);
  }

  int3 cellDim = make_int3(L/rcut)+1;
  if(gcnf.D2) cellDim.z = 1;
    
  real3 cellSize = L/make_real3(cellDim);

  utils.cellDim = cellDim;
  
  utils.invCellSize = 1.0/cellSize;
  if(gcnf.D2) utils.invCellSize.z = real(0.0);
  utils.gridPos2CellIndex = make_int3( 1,
				       utils.cellDim.x,
				       utils.cellDim.x*utils.cellDim.y);
  utils.Lhalf = L*0.5;
  utils.cellSize = cellSize;
}

void NeighbourList_Base::reorderParticles(){
  int nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  int nblocks  =  N/nthreads +  ((N%nthreads!=0)?1:0); 

  NeighbourList::computeHash<NeighbourList::MortonHash><<<nblocks, nthreads>>>(pos.d_m, particleIndex.d_m, particleHash.d_m, N, utils);
  Sorter::sortByKey(particleIndex.d_m, particleIndex_alt.d_m, particleHash, particleHash_alt.d_m, N);
  this->reorderProperty(pos.getTexture(), sortPos.d_m, N);
}

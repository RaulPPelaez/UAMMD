/*Raul P. Pelaez 2017. Cell List implementation
  A neighbour list can compute and provide for each particle a list of particle indices that are closer than a certain distance to it.

  All neighbour list must be capable of:

  -Handling particle groups.
  -Provide an iterator with a list of neighbour particles for each particle
  -Provide an iterator with the number of neighbours of each particle.
  -Provide a "transverse" method that takes a Transverser and goes through every pair of particles with it (See NBodyForces.cuh or RadialPotential.cuh)


  The list for a certain particle i starts at particleOffset[i], after numberNeighbours[i], the neighbour list for particle i contains undefined data.


CellList subdivides the simulation box in cubic cells and uses a hash sort based algorithm to compute a list of particles in each cell. After that, it can compute a verlet list of neighbours or use the cell list itself to transverse particle pairs.





Usage:

   See PairForces.cu for a simple example on how to use a NL.
   Typically transverseList will do a much better job at using the list than asking for it and manually transversing it. But it could be usefull if the list is to be used many times per step. 

TODO:
100- Make a better separation between neighbour list and transverse schemes in this file
100- Improve needsRebuild
 */
#ifndef CELLLIST_CUH
#define CELLLIST_CUH

#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"utils/ParticleSorter.cuh"
#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include"utils/cxx_utils.h"
#include"System/System.h"
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>


#include<cub/cub.cuh>

#include<limits>

namespace uammd{
  namespace CellList_ns{
    constexpr int EMPTY_CELL = std::numeric_limits<int>::max();
    
    //sortPos contains the particle position sorted in a way that ensures that all particles in a cell are one after the other
    //You can identify where each cell starts and end by transversing this vector and searching for when the cell of a particle is  different from the previous one
    template<class InputIterator>
    __global__ void fillCellList(InputIterator sortPos,
				 const int *sortedIndex,
				 int *cellStart, int *cellEnd,
				 int N, Grid grid){
      /*A thread per particle*/
      uint id = blockIdx.x*blockDim.x + threadIdx.x;      
      if(id<N){//If my particle is in range
	
	uint icell, icell2;
	/*Get my icell*/
	icell = grid.getCellIndex(grid.getCell(make_real3(sortPos[id])));
      
	/*Get the previous part.'s icell*/
	if(id>0){ /*Shared memory target VVV*/
	  icell2 = grid.getCellIndex(grid.getCell(make_real3(sortPos[id-1])));
	}
	else
	  icell2 = 0;
	//If my particle is the first or is in a different cell than the previous
	//my i is the start of a cell
	if(id ==0 || icell != icell2){
	  //Then my particle is the first of my cell
	  cellStart[icell] = id;
	
	  //If my i is the start of a cell, it is also the end of the previous
	  //Except if I am the first one
	  if(id>0)
	    cellEnd[icell2] = id;
	}
	//If I am the last particle my cell ends 
	if(id == N-1) cellEnd[icell] = N;      
      }

    }
    /* Multiple threads per particle
    template<int tpp>
    using  WarpScan = cub::WarpScan<int, tpp>;
    template<int tpp, class InputIterator>
    __global__ void fillNeighbourListTPP(InputIterator pos,				      
				      const int *sortedIndex,
				      const int * __restrict__ cellStart, const int * __restrict__ cellEnd,
				      int * __restrict__ neighbourList, int* __restrict__ nNeighbours,
				      int maxNeighbours,
				      real cutOff2,
				      int N, Grid grid,
				      int* tooManyNeighboursFlag){
      int id = (blockIdx.x*blockDim.x + threadIdx.x)/tpp;

      extern __shared__ typename WarpScan<tpp>::TempStorage temp_storage[];

      bool active = true;
      if(id>=N) active = false;
      
      int nneigh = 0;
      int has_neigh = 0;
      int offset = id*maxNeighbours;
      real3 myParticle;
      if(active)
	myParticle = make_real3(pos[id]);
      else
	myParticle = make_real3(0);
      
      int x,y,z;
      int3 cellj;
      int3 celli = grid.getCell(myParticle);
      //Go through all neighbour cells
      //For some reason unroll doesnt help here
      int zi = -1; //For the 2D case
      int zf = 1;
      if(grid.cellDim.z == 1){
	zi = zf = 0;
      }

      const int warpid = threadIdx.x%tpp;
      const int delta = tpp-warpid-1;
      for(x=-1; x<=1; x++){
	cellj.x = grid.pbc_cell_coord<0>(celli.x + x);
	for(z=zi; z<=zf; z++){
	  cellj.z = grid.pbc_cell_coord<2>(celli.z + z);
	  for(y=-1; y<=1; y++){
	    cellj.y = grid.pbc_cell_coord<1>(celli.y + y);	      
	    int icell  = grid.getCellIndex(cellj);
	    //Index of the first particle in the cell's list
	    int firstParticle = cellStart[icell];
	    if(firstParticle != EMPTY_CELL){ //Continue only if there are particles in this cell
	      

	    //Index of the last particle in the cell's list
	      int lastParticle = cellEnd[icell];
	      int nincell = lastParticle-firstParticle;
	    
	      //for(int j=threadIdx.x%tpp; j<nincell; j+=tpp){
	      int j = threadIdx.x%tpp;
	      int first_thread_j;
	      do{
		first_thread_j = j/tpp;
		has_neigh = 0;
		int cur_j;
		if(active && j<nincell){
		  cur_j = sortedIndex[j+firstParticle];
		  if(cur_j != id){
		    real3 pj = make_real3(pos[cur_j]);
		    real3 rij = grid.box.apply_pbc(pj-myParticle);		  
		    if(dot(rij, rij) <= cutOff2){
		      has_neigh = 1;
		    } //endif		  
		  }//endif
		}
		__syncthreads();
		int warp_id = threadIdx.x/tpp;
		int old_nneigh = nneigh;
		int sum;
		int my_offset;		
		WarpScan<tpp>(temp_storage[warp_id]).InclusiveSum(has_neigh, my_offset, sum);
		__syncthreads();
#if __CUDA_ARCH__ > 210
		sum = __shfl_down(my_offset, delta, 32);
#endif
		
		nneigh += sum;
		if(nneigh>=maxNeighbours){
		  atomicMax(tooManyNeighboursFlag, nneigh);
		  return;
		}
		if(has_neigh)
		  neighbourList[offset + old_nneigh + my_offset-1] = cur_j;
		
		j += tpp;
	      }while(first_thread_j<nincell);
	    }
	  }//endfor
	}//endfor
      }//endfor y	  
      if(active && threadIdx.x%tpp == 0)
	nNeighbours[id] = nneigh;

    }
  
*/


    template<class InputIterator>
    __global__ void fillNeighbourList(InputIterator pos,
				      const int *sortedIndex,
				      const int * __restrict__ cellStart, const int * __restrict__ cellEnd,
				      int * __restrict__ neighbourList, int* __restrict__ nNeighbours,
				      int maxNeighbours,
				      real cutOff2,
				      int N, Grid grid,
				      int* tooManyNeighboursFlag){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;
      
      int nneigh = 0;
      const int ori = sortedIndex[id];
      const int offset = ori*maxNeighbours;
      const real3 myParticle = make_real3(pos[id]);
      int3 cellj;
      const int3 celli = grid.getCell(myParticle);
      /**Go through all neighbour cells**/
      //For some reason unroll doesnt help here
      int zi = -1; //For the 2D case
      int zf = 1;
      if(grid.cellDim.z == 1){
	zi = zf = 0;
      }

      for(int x=-1; x<=1; x++){
	cellj.x = grid.pbc_cell_coord<0>(celli.x + x);
	for(int z=zi; z<=zf; z++){
	  cellj.z = grid.pbc_cell_coord<2>(celli.z + z);
	  for(int y=-1; y<=1; y++){
	    cellj.y = grid.pbc_cell_coord<1>(celli.y + y);	      
	    const int icell  = grid.getCellIndex(cellj);
	    
	    /*Index of the first particle in the cell's list*/
	    const int firstParticle = cellStart[icell];
	    if(firstParticle != EMPTY_CELL){ /*Continue only if there are particles in this cell*/
	      /*Index of the last particle in the cell's list*/
	      const int lastParticle = cellEnd[icell];
	      const int nincell = lastParticle-firstParticle;
	    
	      for(int j=0; j<nincell; j++){
		int cur_j = j + firstParticle;// sortedIndex[j+firstParticle];
		if(cur_j != id){
		  real3 pj = make_real3(pos[cur_j]);
		  real3 rij = grid.box.apply_pbc(pj-myParticle);		  
		  if(dot(rij, rij) <= cutOff2){
		    nneigh++;
		    if(nneigh>=maxNeighbours){
		      atomicMax(tooManyNeighboursFlag, nneigh);
		      return;
		    }

		    neighbourList[offset + nneigh-1] = sortedIndex[cur_j];		  
		  } //endif
		}//endif
	      }//endfor
	    }//endif
	  }//endfor y	  
	}//endfor z
      }//endfor x

      nNeighbours[ori] = nneigh;

    }

    //Using a transverser, this kernel processes it by providing it every pair of neighbouring particles    
    template<class Transverser, class InputIterator, class GroupIndexIterator>
    __global__ void transverseCellList(Transverser tr,
  				       InputIterator sortPos,
  				       const int *sortedIndex,
				       GroupIndexIterator groupIndex,
  				       const int * __restrict__ cellStart, const int * __restrict__ cellEnd,
  				       real cutOff2,
  				       int N, Grid grid){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;
   

      /*Initial value of the quantity*/
      auto quantity = tr.zero();

      const int ori = groupIndex[sortedIndex[id]];
      const real4 myParticle = sortPos[id];
      int3 cellj;
      const int3 celli = grid.getCell(myParticle);
      /*Delegator makes possible to call transverseList with a simple Transverser
	(a transverser that only uses positions, with no getInfo method) by using
	SFINAE*/
      /*Note that in the case of a simple Transverser, this code gets trashed by the compiler
	incurring no additional cost. And with inlining, even when T is general no overhead
        comes from Delegator.
      */
      SFINAE::Delegator<Transverser> del;      
      del.getInfo(tr, ori);
      
      /**Go through all neighbour cells**/
      //For some reason unroll doesnt help here
      int zi = -1; //For the 2D case
      int zf = 1;
      if(grid.cellDim.z == 1){
  	zi = zf = 0;
      }
      
      for(int x=-1; x<=1; x++){
  	cellj.x = grid.pbc_cell_coord<0>(celli.x + x);
  	for(int z=zi; z<=zf; z++){
  	  cellj.z = grid.pbc_cell_coord<2>(celli.z + z);
  	  for(int y=-1; y<=1; y++){
  	    cellj.y = grid.pbc_cell_coord<1>(celli.y + y);	      
  	    const int icell  = grid.getCellIndex(cellj);
  	    
  	    /*Index of the first particle in the cell's list*/
  	    const int firstParticle = cellStart[icell];
  	    if(firstParticle != EMPTY_CELL){ /*Continue only if there are particles in this cell*/
  	      /*Index of the last particle in the cell's list*/
  	      const int lastParticle = cellEnd[icell];
  	      const int nincell = lastParticle-firstParticle;
  	    
  	      for(int j=0; j<nincell; j++){
  		int cur_j = j + firstParticle;// sortedIndex[j+firstParticle];
  		if(cur_j != id && cur_j < N){
		  tr.accumulate(quantity, del.compute(tr, groupIndex[sortedIndex[cur_j]], myParticle, sortPos[cur_j]));
  		}//endif
  	      }//endfor
  	    }//endif
  	  }//endfor y	  
  	}//endfor z
      }//endfor x

      tr.set(ori, quantity);
    }

  }


  
  class CellList{
  protected:
    thrust::device_vector<int> cellStart, cellEnd;
    //The neighbour list has the following format: [ neig0_0, neigh1_0,... neighmaxNeigh_0, neigh0_1..., neighmaxNeigh_numberParticles]
    //The list is strided maxNeighboursPerParticle elements between particles, after numberNeighbours[i], the neighbour list for particle i contains undefined data.

    thrust::device_vector<int> neighbourList, numberNeighbours;
    thrust::device_vector<real4> sortPos;
    int maxNeighboursPerParticle;
    int *tooManyNeighboursFlagGPU; //Set to !=0 if a particle has more neighbours than allowed
    ParticleSorter ps;  //hash sort handler

    //My system info
    shared_ptr<ParticleData> pd; 
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;

    //If true, the next issued neighbour list update will always result in a reconstruction of the list.
    bool force_next_update = true;
    bool rebuildNlist;

    //Handles the case where the box or cutoff changed between calls to update
    Box currentBox;
    real currentCutOff;

    
    cudaEvent_t event;
  public:
    //Returns the index of the first neigbour of particle index
    struct NeighbourListOffsetFunctor{
      NeighbourListOffsetFunctor(int a):stride(a){}
      int stride; //This is equal to maxNumberNeighbours
      __host__ __device__ __forceinline__
      int operator()(const int &index) const{
	return index*stride;
      }
    };

    //These aliases describe an iterator that returns the offset of each particle in the list without the need for
    // an array
    using CountingIterator = cub::CountingInputIterator<int>;
    using StrideIterator = cub::TransformInputIterator<int, NeighbourListOffsetFunctor, CountingIterator>;

    //The data structure needed to use the list from outside
    struct NeighbourListData{
      int * neighbourList;
      int *numberNeighbours;
      StrideIterator particleStride = StrideIterator(CountingIterator(0), NeighbourListOffsetFunctor(0));
    };
    
    CellList(shared_ptr<ParticleData> pd,
	     shared_ptr<System> sys): CellList(pd, std::make_shared<ParticleGroup>(pd, sys), sys){
    }

    CellList(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys): pd(pd), pg(pg), sys(sys){
      sys->log<System::MESSAGE>("[CellList] Created");
      
      maxNeighboursPerParticle = 32;
      
      //I want to know if the number of particles has changed
      pd->getNumParticlesChangedSignal()->connect([this](int Nnew){this->handleNumParticlesChanged(Nnew);});      
      
      //The flag has managed memory
      cudaMallocHost(&tooManyNeighboursFlagGPU, sizeof(int), cudaHostAllocMapped);

      //An event to check for the flag during list construction
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    }
    ~CellList(){
      sys->log<System::DEBUG>("[CellList] Destroyed");
      cudaFree(tooManyNeighboursFlagGPU);
    }


    void handleNumParticlesChanged(int Nnew){
      sys->log<System::DEBUG>("[CellList] Number particles changed signal handled.");
      int numberParticles = pg->getNumberParticles();
      if(neighbourList.size()){
	neighbourList.resize(numberParticles*maxNeighboursPerParticle);
	numberNeighbours.resize(numberParticles);
      }
      force_next_update = true;
    }

    NeighbourListData getNeighbourList(cudaStream_t st = 0){
      if(rebuildNlist){
	int numberParticles = pg->getNumberParticles();
	
	neighbourList.resize(numberParticles*maxNeighboursPerParticle);
	numberNeighbours.resize(numberParticles);
      

	rebuildNlist = false;
	//Try to fill the neighbour list until success (the construction will fail if a particle has too many neighbours)
	int3 cellDim = make_int3(currentBox.boxSize/currentCutOff + real(0.5));
	Grid grid(currentBox, cellDim);
      
	int Nthreads=512;
	int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
	auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
	auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
	auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());

	auto groupIndex = pg->getIndexIterator(access::location::gpu);

	int flag;
	do{
	  flag = 0;	  
	  auto neighbourList_ptr = thrust::raw_pointer_cast(neighbourList.data());
	  auto numberNeighbours_ptr = thrust::raw_pointer_cast(numberNeighbours.data());
	  sys->log<System::DEBUG3>("[CellList] fill Neighbour List");	
	  //Multiple threads per particle version
	  // constexpr int tpp = 4;
	  // CellList_ns::fillNeighbourListTPP<tpp><<<Nblocks*tpp,
	  //   Nthreads,
	  //   (tpp/Nthreads)*sizeof(CellList_ns::WarpScan<tpp>::TempStorage),
	  //   st>>>(posGroupIterator,
	  // 	ps.getSortedIndexArray(),
	  // 	cellStart_ptr, cellEnd_ptr,
	  // 	neighbourList_ptr, numberNeighbours_ptr,
	  // 	maxNeighboursPerParticle,
	  // 	cutoff*cutoff,
	  // 	numberParticles,
	  // 	grid,
	  // 	tooManyNeighboursFlagGPU);
	  // thrust::host_vector<int> tmp = neighbourList;
	  // fori(0,10) std::cerr<<tmp[i]<<" ";
	  // std::cerr<<std::endl;
	
	  //For each particle, transverse the 27 neighbour cells using the cell list
	  // and decide if they need to be included in the neighbour list.
	  CellList_ns::fillNeighbourList<<<Nblocks, Nthreads, 0, st>>>(sortPos_ptr, //posGroupIterator,
								       ps.getSortedIndexArray(numberParticles),
								       cellStart_ptr, cellEnd_ptr,
								       neighbourList_ptr, numberNeighbours_ptr,
								       maxNeighboursPerParticle,
								       currentCutOff*currentCutOff,
								       numberParticles,
								       grid,
								       tooManyNeighboursFlagGPU);

	  cudaEventRecord(event, st);
	  cudaEventSynchronize(event);
	
	  flag = *tooManyNeighboursFlagGPU;
	  //If a particle has to many neighbours, increase the maximum neighbours allowed and try again.
	  if(flag != 0){
	    this->maxNeighboursPerParticle += 32;//((flag-maxNeighboursPerParticle)/32+1)*32;	  
	    sys->log<System::DEBUG>("[CellList] Resizing list to %d neighbours per particle",
				    maxNeighboursPerParticle);

	    *tooManyNeighboursFlagGPU = 0;
	    neighbourList.resize(numberParticles*maxNeighboursPerParticle);

	  }
	}while(flag!=0);
      
      }
      auto neighbourList_ptr = thrust::raw_pointer_cast(neighbourList.data());
      auto numberNeighbours_ptr = thrust::raw_pointer_cast(numberNeighbours.data());

      NeighbourListData nl;
      nl.neighbourList = neighbourList_ptr;
      nl.numberNeighbours = numberNeighbours_ptr;
      
      nl.particleStride = StrideIterator(CountingIterator(0),
					 NeighbourListOffsetFunctor(maxNeighboursPerParticle));      
      return nl;

    }

    //Use a transverser to transverse the list using directly the cell list (without constructing a neighbour list)
    template<class Transverser>
    void transverseList(Transverser &tr, cudaStream_t st = 0){
      int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG3>("[CellList] Transversing Cell List with %s", type_name<Transverser>().c_str());

      int3 cellDim = make_int3(currentBox.boxSize/currentCutOff + real(0.5));
      Grid grid(currentBox, cellDim);
      
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
      auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
      auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());

      auto groupIndex = pg->getIndexIterator(access::location::gpu);

      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);
      
      CellList_ns::transverseCellList<<<Nblocks, Nthreads, shMemorySize, st>>>(tr,
						       sortPos_ptr,
						       ps.getSortedIndexArray(numberParticles),
						       groupIndex,
						       cellStart_ptr, cellEnd_ptr,
						       currentCutOff*currentCutOff,
						       numberParticles, grid);

    }
    
    bool needsRebuild(Box box, real cutOff){
      pd->hintSortByHash(box, make_real3(cutOff));
      if(force_next_update){
	//force_next_update = false;
	currentCutOff = cutOff;
	currentBox = box;
	return true;
      }

      if(cutOff != currentCutOff) return true;
      if(box.boxSize.x != currentBox.boxSize.x) return true;
      if(box.boxSize.y != currentBox.boxSize.y) return true;
      if(box.boxSize.z != currentBox.boxSize.z) return true;
      return false;
      
    }
    void updateNeighbourList(Box box, real cutoff, cudaStream_t st = 0){

      if(this->needsRebuild(box, cutoff) == false) return;      
      
      sys->log<System::DEBUG1>("[CellList] Updating list");

      //Get the list parameters
      int numberParticles = pg->getNumberParticles();
      int3 cellDim = make_int3(box.boxSize/cutoff + real(0.5));

      int ncells = cellDim.x*cellDim.y*cellDim.z;

      //Resize if needed
      if(cellStart.size()!= ncells) cellStart.resize(ncells);
      if(cellEnd.size()!= ncells) cellEnd.resize(ncells);
      
      auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
      auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());
      cub::CountingInputIterator<int> it(0);
      
      int Nthreads=512;
      int Nblocks=ncells/Nthreads + ((ncells%Nthreads)?1:0);
      
      //Reset cellStart
      fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(cellStart_ptr, it,
						CellList_ns::EMPTY_CELL, ncells);

      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      
      auto posGroupIterator = pg->getPropertyInputIterator(pos.raw(), access::location::gpu);

      //Sort particle positions by cell morton hash
      ps.updateOrderByCellHash<Sorter::MortonHash>(posGroupIterator,
						   numberParticles,
						   box, cellDim, st);

      //Now the array is sorted by cell hash
      //Reorder the positions to this sorted order
      sortPos.resize(numberParticles);     
      ps.applyCurrentOrder(posGroupIterator, sortPos.begin(), numberParticles, st);
      
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());

      //Fill cell list (cellStart and cellEnd) using the sorted array
      Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      sys->log<System::DEBUG3>("[CellList] fill Cell List");      
      Grid grid(box, cellDim);
      CellList_ns::fillCellList<<<Nblocks, Nthreads, 0, st>>>(sortPos_ptr, //posGroupIterator,
       							      ps.getSortedIndexArray(numberParticles),
       							      cellStart_ptr,
       							      cellEnd_ptr,
       							      numberParticles,
       							      grid);

      rebuildNlist = true;
        
    }
        

  };


}
#endif
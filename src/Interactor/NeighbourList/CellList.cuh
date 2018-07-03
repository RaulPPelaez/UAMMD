/*Raul P. Pelaez 2017. Cell List implementation
  A neighbour list can compute and provide for each particle a list of particle indices that are closer than a certain distance to it.

  All neighbour list must be capable of:

  -Handling particle groups.
  -Provide an iterator with a list of neighbour particles for each particle
  -Provide an iterator with the number of neighbours of each particle.
  -Provide a "transverse" method that takes a Transverser and goes through every pair of particles with it (See NBodyForces.cuh or RadialPotential.cuh)


  The list for a certain particle i starts at particleOffset[i], after numberNeighbours[i], the neighbour list for particle i contains undefined data. i is the index of a particle referred to its group (particle id if the group contains all particles).


CellList subdivides the simulation box in cubic cells and uses a hash sort based algorithm to compute a list of particles in each cell. After that, it can compute a verlet list of neighbours or use the cell list itself to transverse particle pairs.


Usage:

   See PairForces.cu for a simple example on how to use a NL.
   Typically transverseList will do a much better job at using the list than asking for it and manually transversing it. But it could be useful if the list is to be used many times per step. 

   In this case, the function transverseListWithNeighbourList will be the best option. This function creates the verlet list the first time it is called in addition to transverse it. After that each call to it will be ~2 times faster than transverseList. Verlet list creation takes a time similar to a call to transverseList. Use this variant if you are to transverse the list more than 2 times each step.
   

TODO:
100- Make a better separation between neighbour list and transverse schemes in this file
100- Improve needsRebuild (which says yes all the time)
 */
#ifndef CELLLIST_CUH
#define CELLLIST_CUH

#include"ParticleData/ParticleData.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"utils/ParticleSorter.cuh"
#include"utils/Box.cuh"
#include"utils/Grid.cuh"
#include"utils/cxx_utils.h"
#include"utils/TransverserUtils.cuh"
#include"System/System.h"
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>


#include<third_party/cub/cub.cuh>

#include<limits>

namespace uammd{
  namespace CellList_ns{
    constexpr int EMPTY_CELL = std::numeric_limits<int>::max();
    
    //sortPos contains the particle position sorted in a way that ensures that all particles in a cell are one after the other
    //You can identify where each cell starts and end by transversing this vector and searching for when the cell of a particle is  different from the previous one
    template<class InputIterator>
    __global__ void fillCellList(InputIterator sortPos,
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

    //Using a transverser, this kernel processes it by providing it every pair of neighbouring particles
    template<bool is2D, class Transverser, class InputIterator, class GroupIndexIterator>
    __global__ void transverseCellList(Transverser tr,
  				       InputIterator sortPos,
  				       const int *sortedIndex,
				       GroupIndexIterator groupIndex,
  				       const int * __restrict__ cellStart, const int * __restrict__ cellEnd,
  				       int N, Grid grid){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;

      
      //The indices provided to getInfo are order agnostic, they will be the indices of the particles inside the group.
      const int ori = groupIndex[sortedIndex[id]];
#if CUB_PTX_ARCH < 300
      constexpr auto cubModifier = cub::LOAD_DEFAULT;
#else
      constexpr auto cubModifier = cub::LOAD_LDG;
#endif

      cub::CacheModifiedInputIterator<cubModifier, real4> pos_itr(sortPos);
      
      //const real4 myParticle = sortPos[id];
      const real4 myParticle = pos_itr[id];
      const int3 celli = grid.getCell(myParticle);
      /*Delegator makes possible to call transverseList with a simple Transverser
	(a transverser that only uses positions, with no getInfo method) by using
	SFINAE*/
      /*Note that in the case of a simple Transverser, this code gets trashed by the compiler
	incurring no additional cost. And with inlining, even when T is general no overhead
        comes from Delegator.
      */
      //Initial value of the quantity
      auto quantity = tr.zero();
      SFINAE::Delegator<Transverser> del;      
      del.getInfo(tr, ori);

      constexpr int numberNeighbourCells = is2D?9:27;

      for(int i = 0; i<numberNeighbourCells; i++){
	int3 cellj = celli;
	cellj.x += i%3-1;
	cellj.y += (i/3)%3-1;
	if(is2D)
	   cellj.z = 0;
	else
	  cellj.z += i/9-1;
	
	cellj = grid.pbc_cell(cellj);
		
	const int icellj = grid.getCellIndex(cellj);

	  
	const int firstParticle = cellStart[icellj];
	if(firstParticle != EMPTY_CELL){ //Continue only if there are particles in this cell
	  //Index of the last particle in the cell's list
	  const int lastParticle = cellEnd[icellj];	  
	  const int nincell = lastParticle-firstParticle;
	  
	  for(int j=0; j<nincell; j++){
	    int cur_j = j + firstParticle;// sortedIndex[j+firstParticle];
	    if(cur_j < N){
	      tr.accumulate(quantity, del.compute(tr, groupIndex[sortedIndex[cur_j]], myParticle, pos_itr[cur_j]));
	    }//endif
	  }//endfor
	}//endif
      }//endfor
      tr.set(ori, quantity);
    }

    //This can be faster using several threads per particle, I am sure...
    //The neighbourList and nNeighbours arrays store particles indexes in internal sort index,
    // you can change between this order and group order using gid = sortedIndex[id], and between group index and
    // particle id (aka index in global array) using groupIndexIterator[gid].
    //getNeighbourList will return an iterator with group indexes .
    template<bool is2D, class InputIterator>
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
      //const int ori = sortedIndex[id];
      const int offset = id*maxNeighbours;

#if CUB_PTX_ARCH < 300
      constexpr auto cubModifier = cub::LOAD_DEFAULT;
#else
      constexpr auto cubModifier = cub::LOAD_LDG;
#endif

      cub::CacheModifiedInputIterator<cubModifier, real4> pos_itr(pos);

      const real3 myParticle = make_real3(pos_itr[id]);

      const int3 celli = grid.getCell(myParticle);
      /**Go through all neighbour cells**/
      constexpr int numberNeighbourCells = is2D?9:27;

      for(int i = 0; i<numberNeighbourCells; i++){
	int3 cellj = celli;
	cellj.x += i%3-1;
	cellj.y += (i/3)%3-1;
	if(is2D)
	  cellj.z = 0;
	else
	  cellj.z += i/9-1;
	
	cellj = grid.pbc_cell(cellj);
		
	const int icellj = grid.getCellIndex(cellj);
	    
	/*Index of the first particle in the cell's list*/
	const int firstParticle = cellStart[icellj];
	if(firstParticle != EMPTY_CELL){ /*Continue only if there are particles in this cell*/
	  /*Index of the last particle in the cell's list*/
	  const int lastParticle = cellEnd[icellj];
	  const int nincell = lastParticle-firstParticle;
	    
	  for(int j=0; j<nincell; j++){
	    int cur_j = j + firstParticle;// sortedIndex[j+firstParticle];
	    if(cur_j != id){
	      real3 pj = make_real3(pos_itr[cur_j]);
	      real3 rij = grid.box.apply_pbc(pj-myParticle);		  
	      if(dot(rij, rij) <= cutOff2){
		nneigh++;
		if(nneigh>=maxNeighbours){
		  atomicMax(tooManyNeighboursFlag, nneigh);
		  return;
		}

		neighbourList[offset + nneigh-1] = cur_j;		  
	      } //endif
	    }//endif
	  }//endfor
	}//endif
      }//endfor
      //Include self interactions
      neighbourList[offset + nneigh] = id;
      nNeighbours[id] = nneigh+1;

    }
    
    //Applies a transverser to the verlet list
    template<class Transverser, class InputIterator, class GroupIndexIterator>
    __global__ void transverseNeighbourList(Transverser tr,
					    InputIterator sortPos,
					    const int *sortedIndex,
					    GroupIndexIterator groupIndex,
					    int *neighbourList, int *numberNeighbours, int stride,
					    int N, Box Box){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;     
      //The indices provided to getInfo are order agnostic, they will be the indices of the particles inside the group.
      const int gid = sortedIndex[id]; //Index of particle "id" inside the particle group
      const int ori = groupIndex[gid]; //Index of particle "gid" in the global array
#if CUB_PTX_ARCH < 300
      constexpr auto cubModifier = cub::LOAD_DEFAULT;
#else
      constexpr auto cubModifier = cub::LOAD_LDG;
#endif

      cub::CacheModifiedInputIterator<cubModifier, real4> pos_itr(sortPos);
      
      //const real4 myParticle = sortPos[id];
      const real4 myParticle = pos_itr[id];
      /*Delegator makes possible to call transverseList with a simple Transverser
	(a transverser that only uses positions, with no getInfo method) by using
	SFINAE*/
      /*Note that in the case of a simple Transverser, this code gets trashed by the compiler
	incurring no additional cost. And with inlining, even when T is general no overhead
        comes from Delegator.
      */
      //Initial value of the quantity
      auto quantity = tr.zero();
      SFINAE::Delegator<Transverser> del;
      del.getInfo(tr, ori);

      const int nneigh = numberNeighbours[id];

      const int offset = stride*id;
      for(int i = 0; i<nneigh; i++){
	int cur_j = neighbourList[offset+i];
	tr.accumulate(quantity, del.compute(tr, groupIndex[sortedIndex[cur_j]], myParticle, pos_itr[cur_j]));
      }//endfor
      tr.set(ori, quantity);
    }    
  }
  class CellList{
  protected:
    thrust::device_vector<int> cellStart, cellEnd;
    //int3 cellDim;
    Grid grid;
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
    real3 currentCutOff;

    
    cudaEvent_t event;
    
  public:
    
    //Returns the index of the first neigbour of particle index
    struct NeighbourListOffsetFunctor{
      NeighbourListOffsetFunctor(int str, int* sortedIndex):stride(str), sortedIndex(sortedIndex){}
      int stride; //This is equal to maxNumberNeighbours
      int *sortedIndex; //Transforms between internal order and group index
      __host__ __device__ __forceinline__
      int operator()(const int &index) const{
	return sortedIndex[index]*stride;
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
      StrideIterator particleStride = StrideIterator(CountingIterator(0), NeighbourListOffsetFunctor(0, nullptr));
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
      //cudaMallocHost(&tooManyNeighboursFlagGPU, sizeof(int), cudaHostAllocMapped);
      cudaMalloc(&tooManyNeighboursFlagGPU, sizeof(int));
      cudaMemset(tooManyNeighboursFlagGPU, 0, sizeof(real)); 
      //An event to check for the flag during list construction
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    }
    ~CellList(){
      sys->log<System::DEBUG>("[CellList] Destroyed");
      //CudaSafeCall(cudaFreeHost(tooManyNeighboursFlagGPU));
      CudaSafeCall(cudaFree(tooManyNeighboursFlagGPU));
      CudaCheckError();
    }



    //Update the verlet list if necessary and return it
    //NeighbourListData::particleStride will provide particle indexes inside the group, to get particle id (aka index in the global array) use pg->getIndexIterator();
    NeighbourListData getNeighbourList(cudaStream_t st = 0){
      if(currentCutOff.x != currentCutOff.y or
	 currentCutOff.x != currentCutOff.z or
	 currentCutOff.z != currentCutOff.y){
	sys->log<System::CRITICAL>("[CellList] Cannot use NeighbourList with a different cutOff in each direction!. CurrentCutOff: %f %f %f", currentCutOff.x, currentCutOff.y, currentCutOff.z);
      }
      int numberParticles = pg->getNumberParticles();
      if(rebuildNlist){
	
	neighbourList.resize(numberParticles*maxNeighboursPerParticle);
	numberNeighbours.resize(numberParticles);
      

	rebuildNlist = false;
	//Try to fill the neighbour list until success (the construction will fail if a particle has too many neighbours)      
	int Nthreads=128;
	int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
	auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
	auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
	auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());

	//auto groupIndex = pg->getIndexIterator(access::location::gpu);

	int flag;
	do{
	  flag = 0;	  
	  auto neighbourList_ptr = thrust::raw_pointer_cast(neighbourList.data());
	  auto numberNeighbours_ptr = thrust::raw_pointer_cast(numberNeighbours.data());
	  sys->log<System::DEBUG3>("[CellList] fill Neighbour List");		
	  //For each particle, transverse the 27 neighbour cells using the cell list
	  // and decide if they need to be included in the neighbour list.
	  const bool is2D = grid.cellDim.z == 1;
	  if(is2D){
	    CellList_ns::fillNeighbourList<true><<<Nblocks, Nthreads, 0, st>>>(sortPos_ptr, //posGroupIterator,
	   							       ps.getSortedIndexArray(numberParticles),
	   							       cellStart_ptr, cellEnd_ptr,
	   							       neighbourList_ptr, numberNeighbours_ptr,
	   							       maxNeighboursPerParticle,
	   							       currentCutOff.x*currentCutOff.x,
	   							       numberParticles,
	   							       grid,
	   							       tooManyNeighboursFlagGPU);
	  }
	  else{
	    CellList_ns::fillNeighbourList<false><<<Nblocks, Nthreads, 0, st>>>(sortPos_ptr, //posGroupIterator,
								      ps.getSortedIndexArray(numberParticles),
								      cellStart_ptr, cellEnd_ptr,
								      neighbourList_ptr, numberNeighbours_ptr,
								      maxNeighboursPerParticle,
								      currentCutOff.x*currentCutOff.x,
								      numberParticles,
								      grid,
								      tooManyNeighboursFlagGPU);
	  }
	  cudaEventRecord(event, st);
	  cudaEventSynchronize(event);
	
	  //flag = *tooManyNeighboursFlagGPU;
	  cudaMemcpy(&flag, tooManyNeighboursFlagGPU, sizeof(int), cudaMemcpyDeviceToHost);
	  //If a particle has to many neighbours, increase the maximum neighbours allowed and try again.
	  if(flag != 0){
	    this->maxNeighboursPerParticle += 32;//((flag-maxNeighboursPerParticle)/32+1)*32;	  
	    sys->log<System::DEBUG>("[CellList] Resizing list to %d neighbours per particle",
				    maxNeighboursPerParticle);

	    //*tooManyNeighboursFlagGPU = 0;
	    int zero = 0;
	    cudaMemcpy(tooManyNeighboursFlagGPU, &zero, sizeof(int), cudaMemcpyHostToDevice);
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
					 NeighbourListOffsetFunctor(maxNeighboursPerParticle, ps.getSortedIndexArray(numberParticles)));
      return nl;

    }

    
    //Use a transverser to transverse the list using directly the cell list (without constructing a neighbour list)
    template<class Transverser>
    void transverseList(Transverser &tr, cudaStream_t st = 0){
      int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG2>("[CellList] Transversing Cell List with %s", type_name<Transverser>().c_str());
            
      //Grid grid(currentBox, cellDim);

      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
      auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
      auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());

      auto groupIndex = pg->getIndexIterator(access::location::gpu);

      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);

      const bool is2D = grid.cellDim.z == 1;

      if(is2D)
	CellList_ns::transverseCellList<true><<<Nblocks, Nthreads, shMemorySize, st>>>(tr,
						       sortPos_ptr,
						       ps.getSortedIndexArray(numberParticles),
						       groupIndex,
						       cellStart_ptr, cellEnd_ptr,				
						       numberParticles, grid);
      
      else
	CellList_ns::transverseCellList<false><<<Nblocks, Nthreads, shMemorySize, st>>>(tr,
						       sortPos_ptr,
						       ps.getSortedIndexArray(numberParticles),
						       groupIndex,
						       cellStart_ptr, cellEnd_ptr,
						       numberParticles, grid);


    }

    //Use a transverser to transverse the list using the verlet list (constructing a neighbour list if necesary from the cell list)
    template<class Transverser>
    void transverseListWithNeighbourList(Transverser &tr, cudaStream_t st = 0){
      int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG2>("[CellList] Transversing Neighbour List with %s", type_name<Transverser>().c_str());
      //Update verlet list if necesary
      this->getNeighbourList(st);
      int Nthreads=64;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());

      auto groupIndex = pg->getIndexIterator(access::location::gpu);

      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);

      auto neighbourList_ptr = thrust::raw_pointer_cast(neighbourList.data());
      auto numberNeighbours_ptr = thrust::raw_pointer_cast(numberNeighbours.data());

      CellList_ns::transverseNeighbourList<<<Nblocks, Nthreads,
	shMemorySize, st>>>(tr,
			    sortPos_ptr,
			    ps.getSortedIndexArray(numberParticles),
			    groupIndex,
			    neighbourList_ptr, numberNeighbours_ptr, maxNeighboursPerParticle,
			    numberParticles, grid.box);

    }

    bool needsRebuild(Box box, real3 cutOff){
      pd->hintSortByHash(box, cutOff);
      if(force_next_update){
	//force_next_update = false;
	currentCutOff = cutOff;
	currentBox = box;
	return true;
      }

      if(cutOff.x != currentCutOff.x) return true;
      if(cutOff.y != currentCutOff.y) return true;
      if(cutOff.z != currentCutOff.z) return true;
      
      if(box.boxSize.x != currentBox.boxSize.x) return true;
      if(box.boxSize.y != currentBox.boxSize.y) return true;
      if(box.boxSize.z != currentBox.boxSize.z) return true;
      return false;
      
    }
    
    void updateNeighbourList(Box box, real cutOff, cudaStream_t st = 0){
      updateNeighbourList(box, make_real3(cutOff), st);
    }
    
    void updateNeighbourList(Box box, real3 cutOff, cudaStream_t st = 0){

      if(this->needsRebuild(box, cutOff) == false) return;      
      
      sys->log<System::DEBUG1>("[CellList] Updating list");

      //Get the list parameters
      int numberParticles = pg->getNumberParticles();
      grid = Grid(box, cutOff);
      // cellDim = make_int3(box.boxSize/cutoff);
      // if(box.boxSize.z == real(0.0) || cellDim.z == 0) cellDim.z = 1;
      int3 cellDim = grid.cellDim;
      if(cellDim.x < 3) cellDim.x = 3;
      if(cellDim.y < 3) cellDim.y = 3;
      if(box.boxSize.z > real(0.0) && cellDim.z < 3) cellDim.z = 3;

      grid = Grid(box, cellDim);
      sys->log<System::DEBUG1>("[CellList] Using %d %d %d cells", cellDim.x, cellDim.y, cellDim.z);
      uint ncells = grid.getNumberCells(); //cellDim.x*cellDim.y*cellDim.z;

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
						   box, grid.cellDim, st);

      //Now the array is sorted by cell hash
      //Reorder the positions to this sorted order
      sortPos.resize(numberParticles);     
      ps.applyCurrentOrder(posGroupIterator, sortPos.begin(), numberParticles, st);
      
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());

      //Fill cell list (cellStart and cellEnd) using the sorted array
      Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      sys->log<System::DEBUG3>("[CellList] fill Cell List");      
      //Grid grid(box, cellDim);
      CellList_ns::fillCellList<<<Nblocks, Nthreads, 0, st>>>(sortPos_ptr, //posGroupIterator,
       							      cellStart_ptr,
       							      cellEnd_ptr,
       							      numberParticles,
       							      grid);

      rebuildNlist = true;
        
    }

    //These two accesor functions are part of CellList only, not part of the NeighbourList interface.
    const int *getCellStart(){return thrust::raw_pointer_cast(cellStart.data());}
    const int *getCellEnd(){return thrust::raw_pointer_cast(cellEnd.data());}

  protected:
    void handleNumParticlesChanged(int Nnew){
      sys->log<System::DEBUG>("[CellList] Number particles changed signal handled.");
      int numberParticles = pg->getNumberParticles();
      if(neighbourList.size()){
	neighbourList.resize(numberParticles*maxNeighboursPerParticle);
	numberNeighbours.resize(numberParticles);
      }
      force_next_update = true;
    }
  };


  }
#endif




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
  
*/

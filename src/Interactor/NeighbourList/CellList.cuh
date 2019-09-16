/*Raul P. Pelaez 2017. Cell List implementation
  A neighbour list can compute and provide for each particle a list of particle indices that are closer than a certain distance to it.

  All neighbour list must be capable of:

  -Handling particle groups.
  -Provide an iterator with a list of neighbour particles for each particle
  -Provide an iterator with the number of neighbours of each particle.
  -Provide a "transverse" method that takes a Transverser and goes through every pair of particles with it (See NBodyForces.cuh or RadialPotential.cuh)


  The list for a certain particle i starts at particleOffset[i], after numberNeighbours[i], the neighbour list for particle i contains undefined data. i is the index of a particle referred to its group (particle id if the group contains all particles).


CellList subdivides the simulation box in cubic cells and uses a hash sort based algorithm to compute a list of particles in each cell, similar to the algorithm described in [1]. After that, it can compute a verlet list of neighbours or use the cell list itself to transverse particle pairs.


Usage:

 //Create a cellList:
 auto cl = make_shared<CellList>(pd, pg, sys);
 //Update a list
 //If you pass a grid, the cellSize should be >= cutoff
 cl->updateNeighbourList([A box or a grid], cutOff);
 //Traverse the list
 cl->transverseList([A transverser],cudaStream);
 //Get a neighbour list to traverse manually, this will generate the neighbour list whereas traverseList just uses the cell list.
 cl->getNeighbourList();
 //Get a cell list to use manually
 cl->getCellList();


   See PairForces.cu for a simple example on how to use a NL.
   Typically transverseList will do a much better job at using the list than asking for it and manually transversing it. But it could be useful if the list is to be used many times per step. 

   In this case, the function transverseListWithNeighbourList will be the best option. This function creates the verlet list the first time it is called in addition to transverse it. After that each call to it will be ~2 times faster than transverseList. Verlet list creation takes a time similar to a call to transverseList. Use this variant if you are to transverse the list more than 2 times each step.
   

References:

[1] http://developer.download.nvidia.com/assets/cuda/files/particles.pdf
TODO:
100- Make a better separation between neighbour list and transverse schemes in this file

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

    //Using a transverser and the cell list, this kernel processes it by providing it every pair of neighbouring particles
    template<bool is2D, class Transverser>
    __global__ void transverseCellList(Transverser tr,
  				       const real4* sortPos,
  				       const int *groupIndex, //Tranfroms a sortPos index into a group index
				       ParticleGroup::IndexIterator globalIndex, //Transforms a group index into a global index
  				       const int * __restrict__ cellStart, const int * __restrict__ cellEnd,
  				       int N, Grid grid){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;

      
      //The indices provided to getInfo are order agnostic, they will be the indices of the particles inside the group.
      const int ori = globalIndex[groupIndex[id]];
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

      //Go over the first neighbour cells
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
	    int cur_j = j + firstParticle;
	    if(cur_j < N){
	      tr.accumulate(quantity, del.compute(tr, globalIndex[groupIndex[cur_j]], myParticle, pos_itr[cur_j]));
	    }//endif
	  }//endfor
	}//endif
      }//endfor
      tr.set(ori, quantity);
    }

    //This can be faster using several threads per particle, I am sure...
    //The neighbourList and nNeighbours arrays store particles indexes in internal sort index,
    // you can change between this order and group order using gid = groupIndex[id], and between group index and
    // particle id (aka index in global array) using globalIndexIterator[gid].
    //getNeighbourList will return an iterator with group indexes .
    template<bool is2D>
    __global__ void fillNeighbourList(const real4* pos,
				      const int *groupIndex, //Transforms a sortPos index into a group index
				      const int * __restrict__ cellStart, const int * __restrict__ cellEnd,
				      int * __restrict__ neighbourList, int* __restrict__ nNeighbours,
				      int maxNeighbours,
				      real cutOff2,
				      int N, Grid grid,
				      int* tooManyNeighboursFlag){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;
      
      int nneigh = 0;
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
	    int cur_j = j + firstParticle;// groupIndex[j+firstParticle];
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
    template<class Transverser, class InputIterator>
    __global__ void transverseNeighbourList(Transverser tr,
					    InputIterator sortPos,
					    const int *groupIndex, //Transforms a sortPos index into a group index
					    //Transforms a group index into a global index
					    ParticleGroup::IndexIterator globalIndex, 
					    int *neighbourList, int *numberNeighbours, int stride,
					    int N, Box Box){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;     
      //The indices provided to getInfo are order agnostic, they will be the indices of the particles inside the group.
      const int gid = groupIndex[id]; //Index of particle "id" inside the particle group
      const int ori = globalIndex[gid]; //Index of particle "gid" in the global array
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
	tr.accumulate(quantity, del.compute(tr, globalIndex[groupIndex[cur_j]], myParticle, pos_itr[cur_j]));
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
    
    shared_ptr<ParticleData> pd; 
    shared_ptr<ParticleGroup> pg;
    shared_ptr<System> sys;

    //If true, the next issued neighbour list update will always result in a reconstruction of the list.
    bool force_next_update = true;
    bool rebuildNlist;

    //The box and cut off used in the current state of the cell list/neighbour list
    Box currentBox;
    real3 currentCutOff;

    connection numParticlesChangedConnection, posWriteConnection;
    cudaEvent_t event;
    
  public:
    static constexpr auto EMPTY_CELL = CellList_ns::EMPTY_CELL;
    //Returns the index of the first neigbour of particle index
    struct NeighbourListOffsetFunctor{
      NeighbourListOffsetFunctor(int str, int* groupIndex):stride(str), groupIndex(groupIndex){}
      int stride; //This is equal to maxNumberNeighbours
      int *groupIndex; //Transforms between internal order and group index
      __host__ __device__ __forceinline__
      int operator()(const int &index) const{
	return groupIndex[index]*stride;
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
      pd->getPosWriteRequestedSignal()->connect([this](){this->handlePosWriteRequested();});      
      
      //The flag has managed memory
      //cudaMallocHost(&tooManyNeighboursFlagGPU, sizeof(int), cudaHostAllocMapped);
      CudaSafeCall(cudaMalloc(&tooManyNeighboursFlagGPU, sizeof(int)));
      CudaSafeCall(cudaMemset(tooManyNeighboursFlagGPU, 0, sizeof(int))); 
      //An event to check for the flag during list construction
      CudaSafeCall(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    ~CellList(){
      sys->log<System::DEBUG>("[CellList] Destroyed");
      //CudaSafeCall(cudaFreeHost(tooManyNeighboursFlagGPU));
      CudaSafeCall(cudaFree(tooManyNeighboursFlagGPU));
      numParticlesChangedConnection.disconnect();
      posWriteConnection.disconnect();
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
	try{
	  neighbourList.resize(numberParticles*maxNeighboursPerParticle);
	  numberNeighbours.resize(numberParticles);
	}
	catch(thrust::system_error &e){
	  sys->log<System::CRITICAL>("[CellList] Thrust could not resize neighbour list with error: %s", e.what());  
	}
	

	rebuildNlist = false;
	//Try to fill the neighbour list until success (the construction will fail if a particle has too many neighbours)      
	int Nthreads=128;
	int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
	auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
	auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
	auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());

	int flag;
	do{
	  flag = 0;	  
	  auto neighbourList_ptr = thrust::raw_pointer_cast(neighbourList.data());
	  auto numberNeighbours_ptr = thrust::raw_pointer_cast(numberNeighbours.data());
	  sys->log<System::DEBUG3>("[CellList] fill Neighbour List");		
	  //For each particle, transverse the neighbour cells using the cell list
	  // and decide if they need to be included in the neighbour list.
	  const bool is2D = grid.cellDim.z == 1;
	  //Choose the kernel mode
	  auto fillNeighbourList = CellList_ns::fillNeighbourList<false>;	  
	  if(is2D) fillNeighbourList = CellList_ns::fillNeighbourList<true>;
	  
	  fillNeighbourList<<<Nblocks, Nthreads, 0, st>>>(sortPos_ptr,
							  ps.getSortedIndexArray(numberParticles),
							  cellStart_ptr, cellEnd_ptr,
							  neighbourList_ptr, numberNeighbours_ptr,
							  maxNeighboursPerParticle,
							  currentCutOff.x*currentCutOff.x,
							  numberParticles,
							  grid,
							  tooManyNeighboursFlagGPU);
	  CudaSafeCall(cudaEventRecord(event, st));
	  CudaSafeCall(cudaEventSynchronize(event));
	  
	  CudaSafeCall(cudaMemcpy(&flag, tooManyNeighboursFlagGPU, sizeof(int), cudaMemcpyDeviceToHost));
	  //If a particle has to many neighbours, increase the maximum neighbours allowed and try again.
	  if(flag != 0){
	    this->maxNeighboursPerParticle += 32;//((flag-maxNeighboursPerParticle)/32+1)*32;	  
	    sys->log<System::DEBUG>("[CellList] Resizing list to %d neighbours per particle",
				    maxNeighboursPerParticle);

	    //*tooManyNeighboursFlagGPU = 0;
	    int zero = 0;
	    CudaSafeCall(cudaMemcpy(tooManyNeighboursFlagGPU, &zero, sizeof(int), cudaMemcpyHostToDevice));
	    try{
	      neighbourList.resize(numberParticles*maxNeighboursPerParticle);
	    }
	    catch(thrust::system_error &e){
	      sys->log<System::CRITICAL>("[CellList] Thrust could not resize neighbour list with error: %s", e.what());	      
	    }

	  }
	}while(flag!=0);
      
      }
      //If the list does not need rebuild, refill sortPos in case positions have changed
      else{
	auto pos = pd->getPos(access::location::gpu, access::mode::read);     
	auto posGroupIterator = pg->getPropertyInputIterator(pos.raw(), access::location::gpu);
	ps.applyCurrentOrder(posGroupIterator, sortPos.begin(), numberParticles, st);
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
            
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
      auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
      auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());

      auto globalIndex = pg->getIndexIterator(access::location::gpu);

      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);

      const bool is2D = grid.cellDim.z == 1;

      //Choose kernel mode
      auto transverseCellList = CellList_ns::transverseCellList<false, Transverser>;
      if(is2D) transverseCellList = CellList_ns::transverseCellList<true, Transverser>;
      transverseCellList<<<Nblocks, Nthreads, shMemorySize, st>>>(tr,
						       sortPos_ptr,
						       ps.getSortedIndexArray(numberParticles),
						       globalIndex,
						       cellStart_ptr, cellEnd_ptr,				
						       numberParticles, grid);      
      CudaCheckError();
    }

    //Use a transverser to transverse the list using the verlet list (constructing a neighbour list if necesary from the cell list)
    template<class Transverser>
    void transverseListWithNeighbourList(Transverser &tr, cudaStream_t st = 0){
      int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG2>("[CellList] Transversing Neighbour List with %s", type_name<Transverser>().c_str());
      //Update verlet list if necesary
      this->getNeighbourList(st);

      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());

      auto globalIndex = pg->getIndexIterator(access::location::gpu);

      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);

      auto neighbourList_ptr = thrust::raw_pointer_cast(neighbourList.data());
      auto numberNeighbours_ptr = thrust::raw_pointer_cast(numberNeighbours.data());

      CellList_ns::transverseNeighbourList<<<Nblocks, Nthreads,	shMemorySize, st>>>(tr,
			    sortPos_ptr,
			    ps.getSortedIndexArray(numberParticles),
			    globalIndex,
			    neighbourList_ptr, numberNeighbours_ptr, maxNeighboursPerParticle,
			    numberParticles, grid.box);
      CudaCheckError();
    }

    //Check if the cell list needs updating
    bool needsRebuild(Box box, real3 cutOff){
      pd->hintSortByHash(box, cutOff);
      if(force_next_update){
	force_next_update = false;
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

    //Force the cell list to work with a certain grid
    void updateNeighbourList(Grid in_grid, real3 cutOff, cudaStream_t st = 0){
      this->grid = in_grid;
      //Two cells per dimension will cause strange behavior that is probably unexpected
      //If less than 3 cells are to be used it is better to use NBody instead of a cell list
      if(grid.cellDim.x < 3 or
       	 grid.cellDim.y < 3 or
       	 (grid.cellDim.z < 3 and grid.box.boxSize.z != real(0.0))){
       	sys->log<System::CRITICAL>("[CellList] I cannot work with less than 3 cells per dimension!");
      }
      
      //In the case of 3 cells per direction all the particles are checked anyway      
      if(grid.cellDim.x != 3 or
       	 grid.cellDim.y != 3 or
       	 (grid.cellDim.z != 3 and grid.box.boxSize.z != real(0.0))){
	if(grid.cellSize.x < cutOff.x or
	   grid.cellSize.y < cutOff.y or
	   (grid.cellSize.z < cutOff.z and grid.cellSize.z>1)){
	  sys->log<System::CRITICAL>("[CellList] The cell size cannot be smaller than the cut off.");
	}
      }
      if(this->needsRebuild(grid.box, cutOff) == false) return;
      
      sys->log<System::DEBUG1>("[CellList] Updating list");

      //Get the list parameters
      int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG1>("[CellList] Using %d %d %d cells", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
      uint ncells = grid.getNumberCells();

      //Resize if needed
      try{
	if(cellStart.size()!= ncells) cellStart.resize(ncells);
	if(cellEnd.size()!= ncells) cellEnd.resize(ncells);
      }      
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[CellList] Thrust could not resize cell list with error: %s", e.what());	
      }

      auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
      auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());
      cub::CountingInputIterator<int> it(0);
      
      int Nthreads=512;
      int Nblocks=ncells/Nthreads + ((ncells%Nthreads)?1:0);
      
      //Reset cellStart
      fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(cellStart_ptr, it,
						CellList_ns::EMPTY_CELL, ncells);
      CudaCheckError();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      
      auto posGroupIterator = pg->getPropertyInputIterator(pos.raw(), access::location::gpu);

      //Sort particle positions by cell morton hash
      ps.updateOrderByCellHash<Sorter::MortonHash>(posGroupIterator,
						   numberParticles,
						   grid.box, grid.cellDim, st);
      CudaCheckError();
      //Now the array is sorted by cell hash
      //Reorder the positions to this sorted order
      sortPos.resize(numberParticles);     
      ps.applyCurrentOrder(posGroupIterator, sortPos.begin(), numberParticles, st);
      CudaCheckError();
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());

      //Fill cell list (cellStart and cellEnd) using the sorted array
      Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      sys->log<System::DEBUG3>("[CellList] fill Cell List");
      
      CellList_ns::fillCellList<<<Nblocks, Nthreads, 0, st>>>(sortPos_ptr, //posGroupIterator,
       							      cellStart_ptr,
       							      cellEnd_ptr,
       							      numberParticles,
       							      grid);
      CudaCheckError();
      rebuildNlist = true;
    }
    void updateNeighbourList(Box box, real cutOff, cudaStream_t st = 0){
      updateNeighbourList(box, make_real3(cutOff), st);
    }
    void updateNeighbourList(Box box, real3 cutOff, cudaStream_t st = 0){
      
      Grid a_grid = Grid(box, cutOff);
      
       int3 cellDim = a_grid.cellDim;
       if(cellDim.x < 3) cellDim.x = 3;
       if(cellDim.y < 3) cellDim.y = 3;
       if(box.boxSize.z > real(0.0) && cellDim.z < 3) cellDim.z = 3;

       a_grid = Grid(box, cellDim);

      updateNeighbourList(a_grid, cutOff, st);              
    }
    
    //This accesor function is part of CellList only, not part of the NeighbourList interface
    //They allow to obtain a reference to the cell list structures to use them outside
    struct CellListData{
      //[all particles in cell 0, all particles in cell 1,..., all particles in cell ncells]
      //cellStart[i] stores the index of the first particle in cell i (in internal index)
      //cellEnd[i] stores the last particle in cell i (in internal index)
      //So the number of particles in cell i is cellEnd[i]-cellStart[i]
      const int * cellStart, *cellEnd;
      const real4 *sortPos;   //Particle positions in internal index
      const int* groupIndex; //Transformation between internal indexes and group indexes
      Grid grid;
    };
    CellListData getCellList(){
      this->updateNeighbourList(currentBox, currentCutOff);
      CellListData cl;
      try{
	cl.cellStart   =  thrust::raw_pointer_cast(cellStart.data());
	cl.cellEnd     =  thrust::raw_pointer_cast(cellEnd.data());
	cl.sortPos     =  thrust::raw_pointer_cast(sortPos.data());
      }
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[CellList] Thrust could not access cellList arrays with error: %s", e.what());  
      }
      int numberParticles = pg->getNumberParticles();
      cl.groupIndex =  ps.getSortedIndexArray(numberParticles);
      cl.grid = grid;
      return cl;
    }
      
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
    void handlePosWriteRequested(){
      sys->log<System::DEBUG1>("[CellList] Issuing a list update after positions were written to.");
      force_next_update = true;
    }
  };


  }
#endif



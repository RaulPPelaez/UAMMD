/*Raul P. Pelaez 2019. Cell List implementation
  CellList is an instance of the NeighbourList concept.

  CellList subdivides the simulation box in cubic cells and uses a hash sort based algorithm to compute a list of particles in each cell, similar to the algorithm described in [1]. After that, it can compute a verlet list of neighbours or use the cell list itself to trasverse particle pairs.


  A NeighbourList can compute and provide for each particle a list of particle indices that are closer than a certain distance to it.

  There are three ways to interface with NeighbourList:

  1. Providing a Transverser to it
  2. Asking for a NeighbourContainer from it
  3. Asking for a neigbour list from it
  4. Any additional mechanism own to the particular instance (a.i. you can get the cell list binning structure with CellList) 
 
  Methods 1. and 2. are the fastest ones it the list is to be used once/twice per construction, as both methods are not force to construct a neighbour list explicitly.

  Forcing the NeighbourList to construct a neighbour list explicitly with method 3. incurs the overhead of construction, but might be the fastest way if the list is to be used multiple times per construction.

  See usage for instructions on how to use each method.

  All neighbour list must be capable of:

  -Handling particle groups.
  -Provide an iterator with a list of neighbour particles for each particle
  -Provide an iterator with the number of neighbours of each particle.
  -Provide a "transverse" method that takes a Transverser and goes through every pair of particles with it (See NBodyForces.cuh or RadialPotential.cuh)
  -Provide a NeighbourContainer able to provide, for each particle, a forward iterator with its neighbours.*
 
  *Note that this does not imply that a neighbour list is constructed, just that the iterator is able to provide the neighbours.


USAGE:

 //Create a cellList:
 auto cl = make_shared<CellList>(pd, pg, sys);
 //Update a list
 //If you pass a grid, the cellSize should be >= cutoff
 cl->updateNeighbourList([A box or a grid], cutOff);


 //THE DIFFERENT INTERFACE METHODS

 //Traverse the list using the internal CellList mechanism(Method 1.1) 
 cl->transverseList([A transverser],cudaStream);

 //Traverse the list constructing a neighbour list explicitly (Method 1.2)
 //This is similar to Method 3. in that a neighbour list is constructed, but a transverser can be used directly.
 //Method 1.1 will be faster if the list is used once/twice per construction, method 1.2 might be faster otherwise
 cl->transverseListWithNeighbourList([A transverser], cudastream);
  

 //Get a NeighbourContainer (Method 2.). This is equivalent to Method 1.1 (see transverseList kernel), but instead of having to provide a Transverser you can use this structure to get iterators to the neighbours of each particle (as the aforementioned kernel does) and process them manually. This allows for more versatility.
 //See examples/NeighbourListIterator.cu for an example
 auto nc = cl->getNeighbourContainer();


 //Get a neighbour list to traverse manually, this will generate the neighbour list whereas traverseList just uses the cell list. (Method 3.). See more info about this structure below
 auto nl = cl->getNeighbourList();

 

 //Get a cell list to use manually, which provides a spatial binning and a list with the bin of each particle (Method 4.)
 auto cell_list_data = cl->getCellList();


  

Regarding Method 3. 
  The list for a certain particle i starts at nl.particleOffset[i], after nl.numberNeighbours[i], the neighbour list for particle i contains undefined data. i is the index of a particle referred to its group (global id if the group contains all particles).


   See PairForces.cu or examples/NeighbourListIterator.cu for examples on how to use a NL.   

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
    template<class Transverser, class NeighbourContainer>
    __global__ void transverseCellList(Transverser tr,
  				       const real4* sortPos,
  				       const int *groupIndex, //Tranfroms a sortPos index into a group index
				       ParticleGroup::IndexIterator globalIndex, //Transforms a group index into a global index
				       NeighbourContainer ni,
  				       int N){
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id >= N) return;
      
      //The indices provided to getInfo are order agnostic, they will be the indices of the particles inside the group.
      const int ori = globalIndex[groupIndex[id]];
#if CUB_PTX_ARCH < 300
      constexpr auto cubModifier = cub::LOAD_DEFAULT;
#else
      constexpr auto cubModifier = cub::LOAD_LDG;
#endif
      
      const real4 pi = cub::ThreadLoad<cubModifier>(sortPos + id);
      
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
      //for(auto n: ni){
      ni.set(id);
      auto it = ni.begin();
      while(it){
	auto neighbour = *it++;
	const real4 pj = neighbour.getPos();
	const int global_index =  globalIndex[neighbour.getGroupIndex()];
	tr.accumulate(quantity,
		      del.compute(tr,
				  global_index,
				  pi, pj));
      }
      tr.set(ori, quantity);
    }    
    
    //This can be faster using several threads per particle, I am sure...
    //The neighbourList and nNeighbours arrays store particles indexes in internal sort index,
    // you can change between this order and group order using gid = groupIndex[id], and between group index and
    // particle id (aka index in global array) using globalIndexIterator[gid].
    //getNeighbourList will return an iterator with group indexes .
    template<class NeighbourContainer>
    __global__ void fillNeighbourList(const real4* sortPos,
				      const int *groupIndex, //Transforms a sortPos index into a group index
				      NeighbourContainer ni,
				      int * __restrict__ neighbourList, int* __restrict__ nNeighbours,
				      int maxNeighbours,
				      real cutOff2,
				      int N, Box box,
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
      const real3 pi = make_real3(cub::ThreadLoad<cubModifier>(sortPos + id));
      ni.set(id);
      auto it = ni.begin();
      while(it){
	auto n = *it++;
	const int cur_j = n.getInternalIndex();
	const real3 pj = make_real3(cub::ThreadLoad<cubModifier>(sortPos + cur_j));
	const real3 rij = box.apply_pbc(pj-pi);
	if(dot(rij, rij) <= cutOff2){
	  nneigh++;
	  if(nneigh>=maxNeighbours){
	    atomicMax(tooManyNeighboursFlag, nneigh);
	    return;
	  }
	  neighbourList[offset + nneigh-1] = cur_j;		  
	}
      }
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

      const real4 myParticle = cub::ThreadLoad<cubModifier>(sortPos + id);
      auto quantity = tr.zero();
      SFINAE::Delegator<Transverser> del;
      del.getInfo(tr, ori);

      const int nneigh = numberNeighbours[id];

      const int offset = stride*id;
      for(int i = 0; i<nneigh; i++){
	const int cur_j = neighbourList[offset+i];
	const real4 pj = cub::ThreadLoad<cubModifier>(sortPos + cur_j);
	const int global_index = globalIndex[groupIndex[cur_j]];
	tr.accumulate(quantity, del.compute(tr, global_index, myParticle, pj));
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
    
    
  public:
    static constexpr auto EMPTY_CELL = CellList_ns::EMPTY_CELL;
    
    CellList(shared_ptr<ParticleData> pd,
	     shared_ptr<System> sys): CellList(pd, std::make_shared<ParticleGroup>(pd, sys), sys){
    }

    CellList(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys): pd(pd), pg(pg), sys(sys), ps(sys){
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

    
    //The data structure needed to use the neighbour list from outside
    struct NeighbourListData{
      int * neighbourList;
      int *numberNeighbours;
      StrideIterator particleStride = StrideIterator(CountingIterator(0), NeighbourListOffsetFunctor(0, nullptr));
    };

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
	int flag;
	do{
	  flag = 0;	  
	  auto neighbourList_ptr = thrust::raw_pointer_cast(neighbourList.data());
	  auto numberNeighbours_ptr = thrust::raw_pointer_cast(numberNeighbours.data());
	  sys->log<System::DEBUG3>("[CellList] fill Neighbour List");		
	  //For each particle, transverse the neighbour cells using the cell list
	  // and decide if they need to be included in the neighbour list.
	  CellList_ns::fillNeighbourList<<<Nblocks, Nthreads, 0, st>>>(sortPos_ptr,
								       this->getGroupIndexIterator(),
								       this->getNeighbourContainer(),
								       neighbourList_ptr, numberNeighbours_ptr,
								       maxNeighboursPerParticle,
								       currentCutOff.x*currentCutOff.x,
								       numberParticles, grid.box,
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
      // auto cellStart_ptr = thrust::raw_pointer_cast(cellStart.data());
      // auto cellEnd_ptr = thrust::raw_pointer_cast(cellEnd.data());

      auto globalIndex = pg->getIndexIterator(access::location::gpu);

      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<Transverser>().getSharedMemorySize(tr);

      //const bool is2D = grid.cellDim.z == 1;

      //Choose kernel mode
      // auto transverseCellList = CellList_ns::transverseCellList<false, Transverser>;
      // if(is2D) transverseCellList = CellList_ns::transverseCellList<true, Transverser>;
      CellList_ns::transverseCellList<<<Nblocks, Nthreads, shMemorySize, st>>>(tr,
						       sortPos_ptr,
						       this->getGroupIndexIterator(),
						       globalIndex,
									       this->getNeighbourContainer(),
						       numberParticles);      
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
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
      ps.applyCurrentOrder(posGroupIterator, sortPos_ptr, numberParticles, st);
      CudaCheckError();
      

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


    
    class NeighbourContainer; //forward declaration for befriending
  private:
    class NeighbourIterator; //forward declaration for befriending

    //Neighbour is a small accesor for NeighbourIterator
    //Represents a particle, you can ask for its index and position
    struct Neighbour{
      __device__ Neighbour(const Neighbour &other):
      internal_i(other.internal_i){
	groupIndex = other.groupIndex;
	sortPos = other.sortPos;
      }
      //Index in the internal sorted index of the cell list
      __device__ int getInternalIndex(){return internal_i;}
      //Index in the particle group
      __device__ int getGroupIndex(){return groupIndex[internal_i];}
      __device__ real4 getPos(){return cub::ThreadLoad<cub::LOAD_LDG>(sortPos+internal_i);}

    private:
      int internal_i;
      const int* groupIndex;
      const real4* sortPos;
      friend class NeighbourIterator;
      __device__ Neighbour(int i, const int* gi, const real4* sp):
	internal_i(i), groupIndex(gi), sortPos(sp){}
    };

    //This forward iterator must be constructed by NeighbourContainer,
    // Provides a list of neighbours for a certain particle by traversing the neighbouring cells of the particle using the cell list (27 in 3D and 9 in 2D).
    //A neighbour is provided as a Neighbour instance
    class NeighbourIterator:
      public thrust::iterator_adaptor<
 NeighbourIterator,
   int,   Neighbour,
   thrust::any_system_tag,   
   thrust::forward_device_iterator_tag,
   Neighbour,   int
   >{
      friend class thrust::iterator_core_access;
      
      int i; //Particle index
      int j; //Current neighbour index
      CellListData nl;
      
      int ci; //Current cell
      
      int3 celli; //Cell of particle i
      int lastParticle; //Index of last particle in current cell
            
      //Take j to the start of the next cell and return true, if no more cells remain then return false
      __device__ bool nextcell(){
	do{
	  const bool is2D = nl.grid.cellDim.z==1;
	  int3 cellj = celli;
	  cellj.x += ci%3-1;
	  cellj.y += (ci/3)%3-1;
	  cellj.z +=  ci/9-1;
	  
	  cellj = nl.grid.pbc_cell(cellj);
	  const int icellj = nl.grid.getCellIndex(cellj);
	  
	  j = nl.cellStart[icellj];
	  lastParticle = j==EMPTY_CELL?-1:nl.cellEnd[icellj];	  
	  ci++;
	  if(ci >= 9*(is2D?1:3)) return false;
	}while(j == EMPTY_CELL);
	return true;
      }
      //Take j to the next neighbour
      __device__  void increment(){
	if(++j == lastParticle) j = nextcell()?j:-1;	
      }
      __device__ Neighbour dereference() const{
	return Neighbour(j, nl.groupIndex, nl.sortPos);
      }


      //Can only be advanced
      __device__  void decrement() = delete;
      __device__  Neighbour operator[](int i) = delete;
      
      __device__  bool equal(NeighbourIterator const& other) const{
	return other.i == i and other.j==j;
      }

    public:
      //j==-1 means there are no more neighbours and the iterator is invalidated
      __device__  operator bool(){ return j!= -1;}      

    private:
      friend class NeighbourContainer;
      __device__ NeighbourIterator(int i, CellListData nl, bool begin):
	i(i),j(-2), nl(nl), ci(0), lastParticle(-1){
	if(begin){
	  celli = nl.grid.getCell(make_real3(cub::ThreadLoad<cub::LOAD_LDG>(nl.sortPos+i)));
	  increment();
	}
	else j = -1;
      }
    };
  public:
    //This is a pseudocontainer which only purpose is to provide begin() and end() NeighbourIterators for a certain particle
    struct NeighbourContainer{
      int my_i = -1;
      CellListData nl;
      NeighbourContainer(CellListData nl): nl(nl){}
      __device__ void set(int i){this->my_i = i;}
      __device__ NeighbourIterator begin(){return NeighbourIterator(my_i, nl, true);}
      __device__ NeighbourIterator end(){  return NeighbourIterator(my_i, nl, false);}
    };

    NeighbourContainer getNeighbourContainer(){
      auto nl = getCellList();
      return NeighbourContainer(nl);
    }

    
    const real4* getPositionIterator(){
      return thrust::raw_pointer_cast(sortPos.data());
    }
    const int* getGroupIndexIterator(){
      auto nl = getCellList();
      return nl.groupIndex;
    }
    
  };


  }
#endif



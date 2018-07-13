/*Raul P. Pelaez 2018. Adapted from Pablo Ibañez Freire's MonteCarlo Anderson code.

  This module implements Anderson's Monte Carlo NVT GPU algorithm [1].
  The algorithm is presented for hard spheres in two dimensions, but suggests it should be valid for any interparticle potential or dimensionality.

  Works by partitioning the system into a checkerboard like pattern in which the cells of a given subgrid are separated a distance greater or equal than the cut off distance of the potential.
  This allows for the cells of a subgrid to be updated independently. In 3D there are 8 subgrids (4 in 2D).
  Each one of the 8 subgrids is processed sequentially.

  The algorithm can be summarized as follows:
  
  
  1- The checkerboard is placed with a random origin 
  2- For each subgrid perform a certain prefixed number of trials on the particles inside each cell
     
Certain details must be taken into account to ensure detailed-balance:
  1- Any origin in the simulation box must be equally probable
  2- The different subgrids must be processed in a random order
  3- The particles in each cell must be processed in a random order*
  4- Any particle attempt to leave the cell is directly rejected


  * We believe that detailed balance is mantained even if the particles are selected randomly (as opposed to traverse a random permutation of the particle list)
References:

[1] Massively parallel Monte Carlo for many-particle simulations on GPUs. Joshua A. Anderson et. al. https://arxiv.org/pdf/1211.1646.pdf

TODO:
100- Optimize kernel launch parameters and MCStepKernel.
80- Get rid of the CellList, this algorithm can be implemented without reconstructing a cell list from scratch eac step. Although the bottleneck is probably the traversal MCStepKernel.
80- Allow for bonded forces or any other interactor in general.
80- Accept a full-fledged external potential the same way ExternalForces does.

 */

#include"Anderson.cuh"
#include"utils/TransverserUtils.cuh"
#include <thrust/iterator/permutation_iterator.h>
#include<third_party/saruprng.cuh>
namespace uammd {
  namespace MC_NVT{
    template<class Pot,class ExternalPot>
    Anderson<Pot, ExternalPot>::Anderson(shared_ptr<ParticleData> pd,
					 shared_ptr<ParticleGroup> pg,
					 shared_ptr<System> sys,
					 shared_ptr<Pot> pot,
					 shared_ptr<ExternalPot> eP,
					 Parameters in_par):
      pd(pd), pg(pg), sys(sys),
      pot(pot),eP(eP),
      jumpSize(par.initialJumpSize),
      steps(0),      
      par(in_par){

      if(par.kT<real(0.0))
	sys->log<System::CRITICAL>("[MC_NVT::Anderson] Please specify a temperature!");
      
      sys->log<System::MESSAGE>("[MC_NVT::Anderson] Created");
      cl = std::make_shared<CellList>(pd, pg, sys);

      sys->log<System::MESSAGE>("[MC_NVT::Anderson] Temperature: %e", par.kT);
      
      if(par.box.boxSize.z == real(0.0)) this-> is2D = true;
      else is2D = false;
      this->updateSimulationBox(par.box); 
      sys->log<System::MESSAGE>("[MC_NVT::Anderson] Box size: %e %e %e", grid.box.boxSize.x,
				grid.box.boxSize.y,  grid.box.boxSize.z);
      sys->log<System::MESSAGE>("[MC_NVT::Anderson] Grid dimensions: %d %d %d", grid.cellDim.x,
				grid.cellDim.y, grid.cellDim.z);
      
      this->seed = par.seed;
      if(par.seed==0)
	this->seed = sys->rng().next();
      
      cudaStreamCreate(&st);
    }
    
    template<class Pot,class ExternalPot>
    Anderson<Pot,ExternalPot>::~Anderson(){
      cudaStreamDestroy(st);
    }
    
    //This function configures the simulation grid given a box and an interaction CutOff.
    //This process is done in such a way that an even number of cells in each dimension is ensured.
    template<class Pot,class ExternalPot>
    void Anderson<Pot,ExternalPot>::updateSimulationBox(Box box){
	
      this->currentCutOff = make_real3(pot->getCutOff());       

      int3 cellDim = make_int3(box.boxSize/currentCutOff);
      //I need an even number of cells
      if(cellDim.x%2!=0) cellDim.x -= 1;
      if(cellDim.y%2!=0) cellDim.y -= 1; 
      if(cellDim.z%2!=0) cellDim.z -= 1;
      if(is2D){
	cellDim.z = 1;
      }

      if(cellDim.x < 3 or
	 cellDim.y < 3 or
	 cellDim.z == 2)
	sys->log<System::CRITICAL>("[MC_NVT::Anderson] I cannot work with such a large cut off (%e) in this box (%e)!", currentCutOff.x, box.boxSize.x);
      

      this->grid = Grid(box, cellDim);

      //maxOriginDisplacement = sqrt(2)*grid.cellSize.x;
      maxOriginDisplacement = 0.5*grid.box.boxSize.x;
      
      int ncells = grid.getNumberCells();      
     
      triedChanges.resize(ncells);
      acceptedChanges.resize(ncells);
	
      uint* triedChanges_ptr = thrust::raw_pointer_cast(triedChanges.data());
      uint* acceptedChanges_ptr = thrust::raw_pointer_cast(acceptedChanges.data());
	
      int Nthreads=128;
      int Nblocks=ncells/Nthreads + ((ncells%Nthreads)?1:0);      

      fillWithGPU<<<Nblocks, Nthreads>>>(triedChanges_ptr,0, ncells);
      fillWithGPU<<<Nblocks, Nthreads>>>(acceptedChanges_ptr,0, ncells);
    }
    
    //This function computes the systems internal energy (per particle)
    template<class Pot,class ExternalPot>
    real Anderson<Pot,ExternalPot>::computeInternalEnergy(bool resetEnergy){
      sys->log<System::DEBUG>("[MC_NVT::Anderson] Computing total internal energy");
      int numberParticles = pg->getNumberParticles();
      auto globalIndex = pg->getIndexIterator(access::location::gpu);

      // If resetEnergy is true the energy vector is set to 0
      if(resetEnergy){
	auto energy = pd->getEnergy(access::location::gpu, access::mode::write);
	int Nthreads=128;
	int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
	fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(energy.raw(), globalIndex, 0, numberParticles);
      }

      //Energy vector is computed
      cl->updateNeighbourList(grid, currentCutOff);
      auto et = pot->getEnergyTransverser(grid.box, pd);
      cl->transverseList(et, st);
      cudaStreamSynchronize(st);
      //Sum all the energies of the particles in my group
      auto energy = pd->getEnergy(access::location::gpu, access::mode::read);
      auto it = thrust::make_permutation_iterator(energy.raw(), globalIndex);

      real *totalEnergyGPU;
      cudaMalloc(&totalEnergyGPU, sizeof(real));
      {
	size_t newSize = 0;
	cub::DeviceReduce::Sum(nullptr, newSize,
			       it,
			       totalEnergyGPU,
			       numberParticles);
      
	//this check is important because the same storage space is used for several cub calls
	if(newSize > cubTempStorage.size()){
	  cubTempStorage.resize(newSize);	
	}
      }
      size_t size = cubTempStorage.size();
      cub::DeviceReduce::Sum((void*)thrust::raw_pointer_cast(cubTempStorage.data()),
			     size,
			     it,
			     totalEnergyGPU,
			     numberParticles);

      real totalEnergy = 0;
      CudaSafeCall(cudaMemcpy(&totalEnergy, totalEnergyGPU, sizeof(real), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(totalEnergyGPU));
      return 0.5*totalEnergy/numberParticles;
    }

    namespace Anderson_ns{
      //Computes External energy using ExternalPotential and stores it.
      //Can be used with a cub::TransformInputIterator
      template<class ExternalPot>
      struct ExternalEnergyTransform{
	const ExternalPot ePot;
	const real4 *pos;
	real * energy;
	ExternalEnergyTransform(const ExternalPot &ePot,
				const real4 *pos,
				real *energy):
	  ePot(ePot), pos(pos), energy(energy){}
	inline __device__ __host__ real operator()(const int &i) const{
	  real E = ePot.energy(pos[i]);
	  if(energy){
	    energy[i] += E;
	    E = energy[i];
	  }
	  return E;
	}
      };
      
    }
    
    //This function computes the system external energy (per particle)
    template<class Pot,class ExternalPot>
    real Anderson<Pot,ExternalPot>::computeExternalEnergy(bool resetEnergy){
      sys->log<System::DEBUG>("[MC_NVT::Anderson] Computing external energy");
      //If the provided potential has no energy function
      if(SFINAE::has_energy<ExternalPot>::value == false){
	return 0;
      }
      
      int numberParticles = pg->getNumberParticles();
      
      auto globalIndex = pg->getIndexIterator(access::location::gpu);
      
      auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
      
      // If resetEnergy is true energy is set to 0 for particles in the group
      if(resetEnergy){
	int Nthreads=128;
	int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
	fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(energy.raw(), globalIndex, 0, numberParticles);
      }
	    
      auto pos = pd->getPos(access::location::gpu, access::mode::read);		

      cudaStreamSynchronize(st);
      
      real *totalEnergyGPU; cudaMalloc(&totalEnergyGPU, sizeof(real));
      using EnergyTrans = Anderson_ns::ExternalEnergyTransform<ExternalPot>;
      using EnergyComputer = cub::TransformInputIterator<real,
							 EnergyTrans,
							 ParticleGroup::IndexIterator>;

      EnergyComputer energyComputer(globalIndex,
				    EnergyTrans(*eP,
						pos.raw(),
						energy.raw()));      
      
      size_t newSize = 0;
      cub::DeviceReduce::Sum(nullptr, newSize, 
			     energyComputer,
			     totalEnergyGPU,
			     numberParticles);
      
      //this check is important because the same storage space is used for several cub calls
      if(newSize > cubTempStorage.size()){
	cubTempStorage.resize(newSize);	
      }
      //Compute energy and sum it, store each particle's energy as well
      size_t size = cubTempStorage.size();
      cub::DeviceReduce::Sum((void*)thrust::raw_pointer_cast(cubTempStorage.data()),
			     size,
			     energyComputer,
			     totalEnergyGPU,
			     numberParticles);

      real totalEnergy = 0;
      CudaSafeCall(cudaMemcpy(&totalEnergy, totalEnergyGPU, sizeof(real), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(totalEnergyGPU));
      
      return totalEnergy/numberParticles;      
    }
    
    //This functions computes de current acceptance ratio and fits jumpSize in order to obtain the desired one.
    template<class Pot,class ExternalPot>
    void Anderson<Pot,ExternalPot>::updateAccRatio(){
	
      auto tmp = this->getNumberTriesAndNumberAccepted();

      this->resetAcceptanceCounters();
      uint numberTries = tmp.x;
      uint numberAccepted = tmp.y;
      
      //posVar fitting
      const real currentAcceptanceRatio = real(numberAccepted)/numberTries;

      const real3 maxJump = grid.cellSize;
      const real minJumpSize = 0.00001;
      if(currentAcceptanceRatio < par.desiredAcceptanceRatio){
	jumpSize/=par.acceptanceRatioRate;
	if(jumpSize<=minJumpSize) jumpSize=minJumpSize;
      }
      else if(currentAcceptanceRatio > par.desiredAcceptanceRatio){
	jumpSize *= par.acceptanceRatioRate;
	jumpSize = std::min({jumpSize, maxJump.x, maxJump.y});
	if(!is2D){	
	  jumpSize = std::min(jumpSize, maxJump.z);
	}	
      }	
      
      sys->log<System::DEBUG>("[MC_NVT::Anderson] Current acceptance ratio: %e", currentAcceptanceRatio);
      sys->log<System::DEBUG>("[MC_NVT::Anderson] Current step size: %e, %e·cellSize", jumpSize, jumpSize/grid.cellSize.x);
	
    }


    namespace Anderson_ns{
      //Shifts the positions and updates the global position array
      __global__ void upgradeAndShiftKernel(real4* pos,
					    const real4* sortPos,
					    //Transforms between CellList internal index and group index
					    const int *groupIndex,
					    //Transforms between group index and global index
					    ParticleGroup::IndexIterator globalIndex,
					    int N,
					    real3 shift){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=N) return;
	
	pos[globalIndex[groupIndex[i]]] = sortPos[i] + make_real4(shift,0); 

      }
    

      //Shifts the positions in the global position array
      __global__ void shiftKernel(real4* pos,
				  int N,
				  real3 shift){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=N) return;       
	pos[i] += make_real4(shift, 0);
      }
    


    }
    
    //Main Monte Carlo Step
    template<class Pot,class ExternalPot>
    void Anderson<Pot,ExternalPot>::forwardTime(){
      
      sys->log<System::DEBUG>("[MC_NVT::Anderson] Performing Monte Carlo Parallel step: %d",steps);
      steps++;
	
      int numberParticles = pg->getNumberParticles();
      
      currentOrigin = make_real3(0);

      //int dir = sys->rng().next()%3;
      //((real*)(&currentOrigin.x))[dir] = sys->rng().uniform(-grid.cellSize.x*0.5, grid.cellSize.x*0.5);
      
      currentOrigin = make_real3(sys->rng().uniform3(-maxOriginDisplacement, maxOriginDisplacement));
      if(is2D) currentOrigin.z = 0;
      
      sys->log<System::DEBUG1>("[MC_NVT::Anderson] Current origin: %e %e %e",currentOrigin.x, currentOrigin.y, currentOrigin.z);
      
      //Displace the origin of all the particles in the group 
      {
	int Nthreads = 128;
	int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
	    
	auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	    
	Anderson_ns::shiftKernel<<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
							       numberParticles,
							       currentOrigin);
      }
      //Update the cell list
      sys->log<System::DEBUG2>("[MC_NVT::Anderson] Grid size: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
      sys->log<System::DEBUG2>("[MC_NVT::Anderson] Cut off: %e", currentCutOff.x);
      cl->updateNeighbourList(this->grid, this->currentCutOff);

      //Save the sorted positions from the cell list so they can be modified
      sortPos.resize(numberParticles);
      CudaSafeCall(cudaMemcpyAsync(thrust::raw_pointer_cast(sortPos.data()),
				   cl->getCellList().sortPos,
				   numberParticles*sizeof(real4),
				   cudaMemcpyDeviceToDevice,
				   st));
      //Update all subgrids
      if(steps < par.thermalizationSteps){
	step<true>(); //Keep the count of the accepted/rejected movements 
	if(steps%par.tuneSteps == 0)
	  this->updateAccRatio();
      }
      else
	//Just perform the step, without counting accepted/rejected moves
	step<false>();
	
      //Undo shift and copy results to global position array
      {
	int Nthreads = 128;
	int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
	    
	auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
	
	auto groupIndex = cl->getCellList().groupIndex;
	auto globalIndex = pg->getIndexIterator(access::location::gpu);
	    
	Anderson_ns::upgradeAndShiftKernel<<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
									 sortPos_ptr,
									 groupIndex,
									 globalIndex,
									 numberParticles,
									 (-1.0)*currentOrigin);
	currentOrigin = make_real3(0);
      }
      

	
    }

    namespace Anderson_ns{

      //Computes the energy difference in the system due to a particle being in one position (oldPos) or other (newPos)
      template<bool is2D, class PotentialTransverser, class ExternalPot>
      inline __device__ real computeEnergyDifference(int3 celli, //Cell of the particle
						     int selected_i, //CellList index of the selected particle
						     real4 oldPos, //Previous pos
						     real4 newPos, //Moved pos
						     const int *cellStart, const int *cellEnd,
						     Grid grid,
						     const int *groupIndex,
						     const ParticleGroup::IndexIterator &globalIndex,
						     const real4 * sortPos,
						     real3 origin,
						     PotentialTransverser &tr,
						     ExternalPot &eP){


	//Calls ExternalPot::energy if it exists, returns 0 otherwise
	using EpDel = SFINAE::EnergyDelegator<ExternalPot>;
	real newEnergy = EpDel::energy(eP, make_real4(make_real3(newPos) - origin, newPos.w));
	real oldEnergy = EpDel::energy(eP, make_real4(make_real3(oldPos) - origin, oldPos.w));


	//Get any additional info on particle i if needed
	SFINAE::Delegator<PotentialTransverser> del;
	del.getInfo(tr, globalIndex[groupIndex[selected_i]]);
	
	//Sum energy for particles in all neighbouring cells using the new and the old position
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
	  if(firstParticle == CellList_ns::EMPTY_CELL) continue;
	  //Continue only if there are particles in this cell
	  //Index of the last particle in the cell's list
	  const int lastParticle = cellEnd[icellj];	  
	  const int nincell = lastParticle-firstParticle;
	  for(int j=0; j<nincell; j++) {
	    const int cur_j = j + firstParticle;
	    const int global_cur_j = globalIndex[groupIndex[cur_j]];
	    //Compute interaction with both positions
	    real4 posj = sortPos[cur_j];
	    tr.accumulate(oldEnergy, del.compute(tr, global_cur_j, oldPos, posj));
	    //Do not exclude self interactions
	    if(cur_j == selected_i) posj = newPos;
	    tr.accumulate(newEnergy, del.compute(tr, global_cur_j, newPos, posj));
	  
					
	  }
	}
	return newEnergy - oldEnergy;
      }       
	  
      
      //A thread per cell in a 1D thread grid of size ncells/2 for a certain offset in the range [+-1, +-1, +-1]
      //A kernel invocation updates all cells in the subgrid given by offset
      //Each thread will attempt attemptsPerCell movements of particles inside its assigned cell
      //Each thread will update the positions in sortPos
      template<bool countTries, bool is2D,
	       class PotentialTransverser, class ExternalPot>
      __global__ void MCStepKernel(PotentialTransverser tr, ExternalPot eP,
				   const int *cellStart,  const int *cellEnd,
				   int3 offset, //Current subgrid
				   Grid grid,
				   int attemptsPerCell,
				   const int *groupIndex, //Converts between internal CellList index and group index
				   const ParticleGroup::IndexIterator globalIndex, //global index of particles in the group
				   real4* sortPos, //Positions indexed in CellList internal format
				   real3 origin, //Current origin
				   real beta, //1/kT
				   real jumpSize, //Max size of a random particle displacement
				   int step, ullint seed, //RNG seeds
				   uint* tried, uint* accepted, //Per cell acceptance counters
				   int ncells) {

	//Compute my cell
	int3 celli;
	{
	  const int3 cd = grid.cellDim/2;
	  const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	  celli.x = 2*(tid%cd.x) + offset.x;
	  celli.y = 2*((tid/cd.x)%cd.y) + offset.y;
	  celli.z = 2*(tid/(cd.x*cd.y)) + offset.z;
	}
	
	if(is2D)
	  celli.z = 0;
	else if(celli.z>=grid.cellDim.z) return;
	if(celli.x>=grid.cellDim.x or
	   celli.y>=grid.cellDim.y) return;
	
	const int icell = grid.getCellIndex(celli);

	const int firstParticle = cellStart[icell];
	if(firstParticle == CellList_ns::EMPTY_CELL) return;

	const int nincell  = cellEnd[icell] - firstParticle;

	Saru rng(seed, step, icell);

	for(int attempt = 0; attempt < attemptsPerCell; attempt++){
	  tr.zero(); //Maybe the transverser needs to initialize something
	  if(countTries)  tried[icell]++;

	  //Choose a random particle in the cell
	  const int i = firstParticle + int(rng.f()*nincell);
	  //const int i = firstParticle + rng.u32()%nincell;

	  
	  const real4 oldPos = sortPos[i];
	  //Displace with a random vector
	  real4 newPos = oldPos;
	  newPos.x += jumpSize*(real(2.0)*rng.f()-real(1.0));
	  newPos.y += jumpSize*(real(2.0)*rng.f()-real(1.0));
	  if(!is2D)
	    newPos.z += jumpSize*(real(2.0)*rng.f()-real(1.0));	  
	  newPos.w = oldPos.w;

	  //If the attempt takes the particle out of its cell reject it immediately
	  {
	    const int3 newCell = grid.getCell(newPos);
	    if(celli.x != newCell.x or
	       celli.y != newCell.y or
	       celli.z != newCell.z)
	      continue;
	  }
	  //Compute energy different between the two configurations
	  const real dH = computeEnergyDifference<is2D>(celli, i,
						  oldPos, newPos,
						  cellStart, cellEnd,
						  grid,
						  groupIndex, globalIndex,
						  sortPos,
						  origin,
						  tr, eP);
	  //Metropolis acceptance rule	  
	  const real Z = rng.f();
	  const real acceptanceProbabilty = thrust::min(real(1.0), exp(-beta*dH));	  
	  if(Z <= acceptanceProbabilty) {
	    //Move accepted
	    sortPos[i] = newPos;
	    if(countTries) accepted[icell]++;
	  }
    
	} //End attempt loop
      } //End kernel
    

    }


    //This function performs MC steps in each subgrid of the checkerboard
    template<class Pot,class ExternalPot>
    template<bool countTries>
    void Anderson<Pot,ExternalPot>::step(){
      sys->log<System::DEBUG1>("[MC_NVT::Anderson] Launching step kernel with countTries=%d", countTries);
      int numberParticles = pg->getNumberParticles();
		
      auto et = pot->getEnergyTransverser(grid.box, pd);
	
      size_t shMemorySize = SFINAE::SharedMemorySizeDelegator<decltype(et)>().getSharedMemorySize(et);
      
      auto globalIndex = pg->getIndexIterator(access::location::gpu);
	
      uint* triedChanges_ptr = thrust::raw_pointer_cast(triedChanges.data());
      uint* acceptedChanges_ptr = thrust::raw_pointer_cast(acceptedChanges.data());
      	
      real beta = 1.0/par.kT;

      const int numberSubGrids = is2D?4:8;      
      //The different subsets are sorted randomly with Fisher-Yates shuffle
      int shuffled_indexes[] = {0,1,2,3,4,5,6,7};
      fori(0, numberSubGrids-1){	
	int j = i + (sys->rng().next()%(numberSubGrids-i));
	std::swap(shuffled_indexes[i], shuffled_indexes[j]);
      }

      int Nthreads = 128;
      int Nblocks = (grid.getNumberCells()/8)/Nthreads + (((grid.getNumberCells()/8)%Nthreads)?1:0);
      
      auto clData = cl->getCellList();
      auto sortPos_ptr = thrust::raw_pointer_cast(sortPos.data());
      

      //Choose kernel mode
      auto MCStepKernel = &Anderson_ns::MCStepKernel<countTries, false, decltype(et), ExternalPot>;
      if(is2D) MCStepKernel = &Anderson_ns::MCStepKernel<countTries, true, decltype(et), ExternalPot>;
      int ncells = grid.getNumberCells();
      //Go through each subgrid
      for(int i=0; i<numberSubGrids; i++){
	sys->log<System::DEBUG4>("[MC_NVT::Anderson] Launching subgrid %d.", shuffled_indexes[i]);
	MCStepKernel<<<Nblocks, Nthreads, shMemorySize, st>>>
	  (et,*eP,
	   clData.cellStart, clData.cellEnd,
	   offset3D[shuffled_indexes[i]],
	   grid,
	   par.attempsPerCell,
	   clData.groupIndex,
	   globalIndex,
	   sortPos_ptr,
	   currentOrigin,
	   beta,
	   jumpSize,
	   steps, seed,
	   triedChanges_ptr, acceptedChanges_ptr,
	   ncells);
	CudaCheckError();
      }

    }
    
    //This function returns the number of Monte Carlo steps 
    //that have been performed since the last call.
    template<class Pot,class ExternalPot>
    uint2 Anderson<Pot,ExternalPot>::getNumberTriesAndNumberAccepted(){
      int ncells = grid.getNumberCells();
	
      uint* triedChanges_ptr = thrust::raw_pointer_cast(triedChanges.data());
      uint* acceptedChanges_ptr = thrust::raw_pointer_cast(acceptedChanges.data());
	
      tmpStorage.resize(sizeof(uint)*2);

      uint* tmpStorage_ptr = (uint*)thrust::raw_pointer_cast(tmpStorage.data());
      uint* totalTriesGPU = tmpStorage_ptr;
      uint* totalChangesGPU = tmpStorage_ptr+1;

      //cub reduction, the total number of tries and accepted changes is computed
      {
	size_t newSize;
	cub::DeviceReduce::Sum(nullptr, newSize, triedChanges_ptr, totalTriesGPU, ncells, st);
	//this check is important because the same storage space is used for several cub calls
	if(newSize > cubTempStorage.size()){
	  cubTempStorage.resize(newSize);	
	}
      }
      
      void* cubTempStorage_ptr = (void*)thrust::raw_pointer_cast(cubTempStorage.data());
      size_t cubTempStorageSize = cubTempStorage.size();


      //cub reduction, the total number of tries and accepted changes is computed
      cub::DeviceReduce::Sum(cubTempStorage_ptr, cubTempStorageSize,
			     triedChanges_ptr, totalTriesGPU, ncells, st);
      cub::DeviceReduce::Sum(cubTempStorage_ptr, cubTempStorageSize,
			     acceptedChanges_ptr, totalChangesGPU, ncells, st);

      CudaSafeCall(cudaMemcpy(&totalTries, totalTriesGPU, sizeof(int), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemcpy(&totalChanges, totalChangesGPU, sizeof(int), cudaMemcpyDeviceToHost));
      
      return {totalTries, totalChanges};
    }

    template<class Pot,class ExternalPot>
    void  Anderson<Pot,ExternalPot>::resetAcceptanceCounters(){

      int ncells = grid.getNumberCells();
	
      uint* triedChanges_ptr = thrust::raw_pointer_cast(triedChanges.data());
      uint* acceptedChanges_ptr = thrust::raw_pointer_cast(acceptedChanges.data());

      int Nthreads=128;
      int Nblocks=ncells/Nthreads + ((ncells%Nthreads)?1:0);
      
      //Reset
      fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(triedChanges_ptr, 0, ncells);
      fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(acceptedChanges_ptr, 0, ncells);

    }
    
  }
}

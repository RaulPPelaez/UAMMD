/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs

  See BondedForces.cuh for more info.

TODO:
100 - Implement a warp reduction in computeForces
100- Implement sumEnergy and sumVirial
*/

#include"BondedForces.cuh"
#include<cub/cub.cuh>
#include<thrust/sort.h>
#include<thrust/reduce.h>
#include<algorithm>
#include<cstdint>
#include<vector>
#include<set>
#include<fstream>
namespace uammd{
  template<class BondType>
  BondedForces<BondType>::BondedForces(shared_ptr<ParticleData> pd,
				       shared_ptr<System> sys,
				       Parameters par,
				       BondType bondForce):
    Interactor(pd,
	       std::make_shared<ParticleGroup>(pd, sys, "All"),
	       sys,
	       "BondedForces/" + type_name<BondType>()),
    bondForce(bondForce), TPP(64){
    
    //BondedForces does not care about any parameter update, but the BondType might.
    this->setDelegate(&(this->bondForce));

    int numberParticles = pg->getNumberParticles();
    
    sys->log<System::MESSAGE>("[BondedForces] Initialized");

    sys->log<System::MESSAGE>("[BondedForces] Using: %s", type_name<BondType>().c_str());
    nbonds = nbondsFP = 0;
    /*If some bond type number is zero, the loop will simply not be entered, and no storage will be used*/
    /*Read the bond list from the file*/
    std::ifstream in(par.file);
    if(!in.good()){
      sys->log<System::CRITICAL>("[BondedForces] Bond file %s not found!!", par.file.c_str());
    }
    in>>nbonds;
    if(nbonds>0){
      bondList.resize(nbonds*2);//Allocate 2*nbonds, see init for explication
      thrust::host_vector<Bond> h_bondList = bondList;
      fori(0, nbonds){
	if(in.eof()){
	  sys->log<System::CRITICAL>("[BondedForces] ERROR! Bond file ended too soon! Expected %d lines, found %d", nbonds, i);
	}
	in>>h_bondList[i].i>>h_bondList[i].j;
	if(h_bondList[i].i >= numberParticles or h_bondList[i].j >= numberParticles)
	  sys->log<System::WARNING>("[BondedForces] Bond %d involves particles with index beyond the total number of particles!. i: %d, j:%d, N: %d", i, h_bondList[i].i, h_bondList[i].j, numberParticles);
	h_bondList[i].bond_info = BondType::readBond(in);
      }
      bondList = h_bondList;
    }
    sys->log<System::MESSAGE>("[BondedForces] %d particle-particle bonds detected.", bondList.size()/2);
    /*Fixed point bonds*/
    if(!in) nbondsFP = 0;
    else in>>nbondsFP;
    if(nbondsFP>0){
      bondListFP.resize(nbondsFP);
      thrust::host_vector<BondFP> h_bondListFP = bondListFP;
      fori(0, nbondsFP){
	in>>h_bondListFP[i].i;
	if(h_bondListFP[i].i >= numberParticles)
	  sys->log<System::WARNING>("[BondedForces] Bond %d involves a particle with index beyond the total number of particles!. i: %d, N: %d", i, h_bondListFP[i].i, numberParticles);
	in>>h_bondListFP[i].pos.x>>h_bondListFP[i].pos.y>>h_bondListFP[i].pos.z;
	h_bondListFP[i].bond_info = BondType::readBond(in);
      }
      bondListFP = h_bondListFP;
    }

    sys->log<System::MESSAGE>("[BondedForces] Detected: %d particle-particle bonds and %d Fixed Point bonds",
			      bondList.size()/2, bondListFP.size());

    /*Upload and init GPU*/
    init();  
  }


  template<class BondType>
  BondedForces<BondType>::~BondedForces(){
    cudaDeviceSynchronize();
    sys->log<System::MESSAGE>("[BondedForces] Destroyed");
  }

  namespace BondedForces_ns{
    //Criterion to sort bonds

    template<class BondType>
    struct BondComp{
      __device__ __host__ bool operator()(const BondType &a, const BondType &b){
	return a.i<b.i;
      }
    };

    template<class Bond>
    //Takes a bondList that is filled from 0 to nbonds and mirrors it in nbonds 2nbonds
    __global__ void dupicateBonds(Bond * bondList,
				  int nbonds){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id >= nbonds) return;
      int i = id + nbonds;
      Bond b = bondList[i-nbonds];
      thrust::swap(b.i, b.j);
      bondList[i] = b;
    }


  }


  //Initialize variables and upload them to GPU, init CUDA
  template<class BondType>
  void BondedForces<BondType>::init(){
    if(nbonds > 0)
      this->initParticleParticle();
    if(nbondsFP > 0)
      this->initFixedPoint();
    sys->log<System::MESSAGE>("[BondedForces] %d particles have at least one bond",
			      std::max(bondStart.size(), bondListFP.size()));
  }

  template<class BondType>
  void BondedForces<BondType>::initParticleParticle(){
    int numberParticles = pg->getNumberParticles();
    // ****************************************Pair bonds********************************************* 
    //This algorithm is identical to the one used in PairForces to sort by cell
    //First store all bonded pairs. That means i j and j i
    //The first ones are readed given, the complementary have to be generated
    
    int BLOCKSIZE = 128;
    int Nthreads = BLOCKSIZE<nbonds?BLOCKSIZE:nbonds;
    int Nblocks  =  nbonds/Nthreads +  ((nbonds%Nthreads!=0)?1:0); 

    sys->log<System::DEBUG>("[BondedForces] Duplicating bonds");
    auto d_bondList = thrust::raw_pointer_cast(bondList.data());
    
    BondedForces_ns::dupicateBonds<<<Nblocks, Nthreads>>>(d_bondList, nbonds);

    sys->log<System::DEBUG>("[BondedForces] Sorting bonds");
    /*Sort in the i index to construct bondStart and bondEnd*/
    thrust::sort(bondList.begin(), bondList.end(), BondedForces_ns::BondComp<Bond>());
    
    nbonds = bondList.size();
    //We have a list of bonds ordered by its first particle, so; All the particles
    // bonded with particle i=0, all particles "" i=1...

    //We need additional arrays to know where in the list the bonds of particle i start
    // and end
    //Initially all bondStarts are 2^32-1, this value means no particles bonded
    sys->log<System::DEBUG>("[BondedForces] Computing number of particles with bonds");
    thrust::host_vector<Bond> h_bondList = bondList;
    std::set<int> particlesWithBonds;
    fori(0, h_bondList.size()){
      particlesWithBonds.insert(h_bondList[i].i);
    }
    
    int nParticlesWithBonds = particlesWithBonds.size();
    sys->log<System::DEBUG>("[BondedForces] %d particles with bonds found", nParticlesWithBonds);
    
    bondStart.resize(nParticlesWithBonds, nullptr);
    nbondsPerParticle.resize(nParticlesWithBonds, 0);

    thrust::host_vector<Bond*> h_bondStart = bondStart;
    thrust::host_vector<int> h_nbondsPerParticle = nbondsPerParticle;
    sys->log<System::DEBUG>("[BondedForces] Filling bondStart");

    //Fill helper data structures
    int index = 0;
    int nbondsi = 0;
    for(int b = 0; b<nbonds; b++){
      int i = h_bondList[b].i;
      int inext;
      if(b<nbonds-1) inext = h_bondList[b+1].i;
      else inext = -1;
      nbondsi++;
      if(inext != i){
	if(index == 0)
	  h_bondStart[0] = thrust::raw_pointer_cast(bondList.data());    
	else
	  h_bondStart[index] = thrust::raw_pointer_cast(bondList.data())+b+1-nbondsi;
	
	h_nbondsPerParticle[index] = nbondsi;

	index++;
	nbondsi = 0;
      }
    }
    bondStart = h_bondStart;
    nbondsPerParticle = h_nbondsPerParticle;
    int meanBondsPerParticle = thrust::reduce(nbondsPerParticle.begin(), nbondsPerParticle.end())/bondStart.size();
    TPP = std::min((meanBondsPerParticle/32)*32, 128);
    TPP = std::max(TPP, 32);
    sys->log<System::MESSAGE>("[BondedForces] Mean bonds per particle: %d", meanBondsPerParticle);
    sys->log<System::DEBUG>("[BondedForces] Using %d threads per particle", TPP);
        
  }

  template<class BondType>
  void BondedForces<BondType>::initFixedPoint(){
    int numberParticles = pg->getNumberParticles();    
    sys->log<System::DEBUG>("[BondedForces] Sorting fixed point bonds");
    //Sort in the i index to construct bondStart and bondEnd
    thrust::sort(bondListFP.begin(), bondListFP.end(), BondedForces_ns::BondComp<BondFP>());

    sys->log<System::DEBUG>("[BondedForces] Computing number of particles with bonds");
    thrust::host_vector<BondFP> h_bondList = bondListFP;
    std::set<int> particlesWithBonds;
    fori(0, h_bondList.size()){
      particlesWithBonds.insert(h_bondList[i].i);
    }
    
    int nParticlesWithBonds = particlesWithBonds.size();
    sys->log<System::DEBUG>("[BondedForces] %d particles with fixed point bonds found", nParticlesWithBonds);
    
    bondStartFP.resize(nParticlesWithBonds, nullptr);
    nbondsPerParticleFP.resize(nParticlesWithBonds, 0);
      
    sys->log<System::DEBUG>("[BondedForces] Filling bondStartFP");

    //Fill helper data structures
    int index = 0;
    int nbondsi = 0;
    for(int b = 0; b<nbondsFP; b++){
      int i = h_bondList[b].i;
      int inext;
      if(b<nbondsFP-1) inext = h_bondList[b+1].i;
      else inext = -1;
      nbondsi++;
      if(inext != i){
	if(index == 0)
	  bondStartFP[0] = thrust::raw_pointer_cast(bondListFP.data());    
	else
	  bondStartFP[index] = thrust::raw_pointer_cast(bondListFP.data())+b+1-nbondsi;
	
	nbondsPerParticleFP[index] = nbondsi;

	index++;
	nbondsi = 0;
      }
    }
    }

  namespace BondedForces_ns{

    //I do not really like how this is written now, but it really improves performance...
    
    //This version assigns a block for each particle (thread threadIdx.x handles the bond threadIdx.x of particle blockIdx.x) Much faster when particles have many bonds per particle (>32 maybe)
    template<class Bond, class BondType, int THREADS_PER_BLOCK>
    __global__ void computeBondedForcesBlockPerParticle(real4* __restrict__ force, const real4* __restrict__ pos,
					Bond** __restrict__ bondStart,
					const int* __restrict__ nbondsPerParticle,
					BondType bondForce,
					const int * __restrict__ id2index){
      //This little trick of unionizing the shared memory for the block shared parameters and the blockreduce storage      // does not seem to help much, at least in a GTX980.
      struct Shared{
	Bond const * bondList;
	int nbonds;
	int p;
	real3 posi;
      };
      using BlockReduce = cub::BlockReduce<real3, THREADS_PER_BLOCK>;
      __shared__ union{
	Shared info;
	typename BlockReduce::TempStorage temp_storage;
      } shared;
       
      __shared__ int index;

      //Bond list for my particle
      if(threadIdx.x == 0){
	shared.info.bondList = bondStart[blockIdx.x];
	//Number of bonds for my particle
        shared.info.nbonds = nbondsPerParticle[blockIdx.x];
	//My particle index
        shared.info.p = shared.info.bondList[0].i;
	index = id2index[shared.info.p];
	shared.info.posi = make_real3(pos[index]);
      }

      //A block per particle
      //Instead of launching a thread per particle and discarding those without any bond,
      //I store an additional array of size N_particles_with_bonds that contains the indices
      //of the particles that are involved in at least one bond. And only launch N_particles_with_bonds blocks
      
      //Each thread in a block computes the force on particle p due to one (or several) bonds      

      //My local force accumulator
      real3 f = make_real3(real(0.0));
      
      __syncthreads();    
      for(int b = threadIdx.x; b<shared.info.nbonds; b += blockDim.x){
	
	//Read bond info
	const auto bond = shared.info.bondList[b];
	//Bring pos of other particle
	const real3 posj = make_real3(pos[id2index[bond.j]]);
    
	//Compute force
	real3 r12 =  shared.info.posi-posj;
	//Sum force 
	f += bondForce.force(shared.info.p, bond.j, r12, bond.bond_info);
      }

      //Sum the forces of all threads in my block
      real3 ft;
      if(threadIdx.x < shared.info.nbonds){
	ft = BlockReduce(shared.temp_storage).Sum(f);	
      }
      __syncthreads();
      //First thread writes to memory
      if(threadIdx.x == 0){
	force[index] += make_real4(ft);
      }
      
    }


    //This version assigns a thread for each particle (thread i handles all the bonds of particle i), works well when particles have a low number of bonds (<32 per particle)
    template<class Bond, class BondType>
    __global__ void computeBondedForcesThreadPerParticle(real4* __restrict__ force, const real4* __restrict__ pos,
					   Bond** __restrict__ bondStart,
					   const int* __restrict__ nbondsPerParticle,
					   BondType bondForce,
					   const int * __restrict__ id2index, int N){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;
      
      auto bondList = bondStart[id];
      //Number of bonds for my particle
      const int nbonds = nbondsPerParticle[id];
      //My particle index
      const int p = bondList[0].i;
      const int index = id2index[p];
      const real3 posi = make_real3(pos[index]);
            
      real3 f = make_real3(real(0.0));
      
      for(int b = 0; b<nbonds; b++){
	
	//Read bond info
	const auto bond = bondList[b];
	//Bring pos of other particle
	const real3 posj = make_real3(pos[id2index[bond.j]]);
    
	//Compute force
	real3 r12 =  posi-posj;
      
	f += bondForce.force(p, bond.j, r12, bond.bond_info);
      }
   
      force[index] += make_real4(f);	
    }


    //The same approach could be used for Fixed Point bonds as with p-p bonds.
    template<class Bond, class BondType>
    __global__ void computeBondedForcesFixedPoint(real4* __restrict__ force,
						  const real4* __restrict__ pos,
						  Bond** __restrict__ bondStart,
						  const int* __restrict__ nbondsPerParticle,
						  BondType bondForce,
						  const int * __restrict__ id2index){
      extern __shared__ char shMem[];      
      real3 *forceTotal = (real3*) shMem; //Each thread has a force
      /*
      real4 &posi = *((real4*)(shMem+blockDim.x*sizeof(real4)));
      int &nbonds = *((int*)&posi + sizeof(real4));
      int &p = *((int*)&nbonds + sizeof(int));
      Bond* &bondList = *((Bond**)&p+sizeof(int));
      */
      
      //Bond list for my particle
      const Bond *bondList = bondStart[blockIdx.x];
      //Number of bonds for my particle
      const int nbonds = nbondsPerParticle[blockIdx.x];
      //My particle index
      const int p = bondList[0].i;
      const real3 posi = make_real3(pos[id2index[p]]);

      //A block per particle
      //Instead of launching a thread per particle and discarding those without any bond,
      //I store an additional array of size N_particles_with_bonds that contains the indices
      //of the particles that are involved in at least one bond. And only launch N_particles_with_bonds blocks
      
      //Each thread in a block computes the force on particle p due to one (or several) bonds
      
            
      real3 f = make_real3(real(0.0));          


      //__syncthreads();    
      for(int b = threadIdx.x; b<nbonds; b += blockDim.x){
	
	//Read bond info
	auto bond = bondList[b];	
    
	//Compute force
	real3 r12 =  posi - bond.pos;
      
        

	f += bondForce.force(p,-1, r12, bondList[b].bond_info);
      }

      /*The first thread sums all the contributions*/
      forceTotal[threadIdx.x] = f;
      __syncthreads();
      //TODO Implement a warp reduction
      if(threadIdx.x==0){
	real3 ft = make_real3(0.0f);
	for(int i=0; i<blockDim.x; i++){
	  ft += forceTotal[i];
	}
	/*Write to global memory*/
	force[id2index[p]] += make_real4(ft);
      }

    }     
  }


  //This function chooses which version of the kernel computeBondedForces to use as a function of the number of bonds per particle
  template<class BondType>
  void BondedForces<BondType>::callComputeBondedForces(cudaStream_t st){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite);

    auto id2index = pd->getIdOrderedIndices(access::location::gpu);
      
    auto d_bondStart = thrust::raw_pointer_cast(bondStart.data());
    auto d_nbondsPerParticle = thrust::raw_pointer_cast(nbondsPerParticle.data());
    uint Nparticles_with_bonds = bondStart.size();

    
    if(TPP<=32 || Nparticles_with_bonds < 5000){ //Empirical magic numbers, could probably be chosen better
      int Nthreads= 128;
      int Nblocks=Nparticles_with_bonds/Nthreads + ((Nparticles_with_bonds%Nthreads)?1:0);

      BondedForces_ns::computeBondedForcesThreadPerParticle<Bond, BondType>
	<<< Nblocks, Nthreads, 0, st>>>(
	      force.raw(), pos.raw(),
	      d_bondStart, d_nbondsPerParticle,
	      bondForce,
	      id2index, Nparticles_with_bonds);
    }
    else{
      //This is due to cub having blocksize as template parameter, I hate it
      if(TPP>=128){
	BondedForces_ns::computeBondedForcesBlockPerParticle<Bond, BondType, 128>
	  <<<Nparticles_with_bonds, 128, 0, st>>>(
		force.raw(), pos.raw(),
		d_bondStart, d_nbondsPerParticle,
		bondForce, id2index);
      }
      else{
	BondedForces_ns::computeBondedForcesBlockPerParticle<Bond, BondType, 64>
	  <<<Nparticles_with_bonds, 64, 0, st>>>(
	        force.raw(), pos.raw(),
		d_bondStart, d_nbondsPerParticle,
		bondForce, id2index);
      }

    }
  }
  template<class BondType>
  void BondedForces<BondType>::sumForce(cudaStream_t st){
    sys->log<System::DEBUG1>("[BondedForces] Computing Forces...");
    if(nbonds>0){
      sys->log<System::DEBUG3>("[BondedForces] Computing Particle-Particle...");
      this->callComputeBondedForces(st);
    }
    if(nbondsFP>0){
      sys->log<System::DEBUG3>("[BondedForces] Computing Fixed-Point...");
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto force = pd->getForce(access::location::gpu, access::mode::readwrite);

      auto id2index = pd->getIdOrderedIndices(access::location::gpu);
      
      int numberParticlesWithBonds = bondStartFP.size();
      auto d_bondStart = thrust::raw_pointer_cast(bondStartFP.data());
      auto d_nbondsPerParticle = thrust::raw_pointer_cast(nbondsPerParticleFP.data());      
      BondedForces_ns::computeBondedForcesFixedPoint
	<<<numberParticlesWithBonds,	TPP, TPP*sizeof(real3), st>>>(
								      force.raw(), pos.raw(),
								      d_bondStart,
								      d_nbondsPerParticle,
								      bondForce,
								      id2index);
    }

  
  }
  template<class BondType>
  real BondedForces<BondType>::sumEnergy(){
    return 0;
  }

}
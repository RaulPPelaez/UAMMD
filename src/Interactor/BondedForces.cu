/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs

  See BondedForces.cuh for more info.

TODO:
100 - Implement a warp reduction in computeForces
100- Implement sumEnergy and sumVirial
*/

#include"BondedForces.cuh"

#include<thrust/sort.h>
#include<thrust/reduce.h>
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
      sys->log<System::CRITICAL>("[BondedForces] Bond file %s not found!!", par.file);
    }
    in>>nbonds;
    if(nbonds>0){
      bondList.resize(nbonds*2);//Allocate 2*nbonds, see init for explication
      thrust::host_vector<Bond> h_bondList = bondList;
      fori(0, nbonds){
	in>>h_bondList[i].i>>h_bondList[i].j;
	h_bondList[i].bond_info = BondType::readBond(in);
      }
      bondList = h_bondList;
    }
    sys->log<System::MESSAGE>("[BondedForces] %d particle-particle bonds detected.", bondList.size()/2);
    /*Fixed point bonds*/
    in>>nbondsFP;
    if(nbondsFP>0){
      bondListFP.resize(nbondsFP);
      thrust::host_vector<BondFP> h_bondListFP = bondListFP;
      fori(0, nbondsFP){
	in>>h_bondListFP[i].i;
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
	  bondStart[0] = thrust::raw_pointer_cast(bondList.data());    
	else
	  bondStart[index] = thrust::raw_pointer_cast(bondList.data())+b+1-nbondsi;
	
	nbondsPerParticle[index] = nbondsi;

	index++;
	nbondsi = 0;
      }
    }

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
  
    template<class Bond, class BondType>
    __global__ void computeBondedForces(real4* __restrict__ force, const real4* __restrict__ pos,
					Bond** __restrict__ bondStart,
					const int* __restrict__ nbondsPerParticle,
					BondType bondForce,
					const int * __restrict__ id2index){
      extern __shared__ char shMem[];
      
      real4 *forceTotal = (real4*) shMem; //Each thread has a force

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
      
            
      real4 f = make_real4(real(0.0));
          
      //__syncthreads();    
      for(int b = threadIdx.x; b<nbonds; b += blockDim.x){
	
	//Read bond info
	const auto bond = bondList[b];
	const int j = bond.j;
	//Bring pos of other particle
	const real3 posj = make_real3(pos[id2index[j]]);
    
	//Compute force
	real3 r12 =  posi-posj;
      
	const real fmod = bondForce.force(p, j, r12, bondList[b].bond_info);

	f += make_real4(fmod*r12);

      }

      /*The first thread sums all the contributions*/
      forceTotal[threadIdx.x] = f;
      __syncthreads();
      //TODO Implement a warp reduction
      if(threadIdx.x==0){
	real4 ft = make_real4(0.0f);
	for(int i=0; i<blockDim.x; i++){
	  ft += forceTotal[i];
	}
	/*Write to global memory*/
	force[id2index[p]] += ft;
      }

    }
    
    template<class Bond, class BondType>
    __global__ void computeBondedForcesFixedPoint(real4* __restrict__ force,
						  const real4* __restrict__ pos,
						  Bond** __restrict__ bondStart,
						  const int* __restrict__ nbondsPerParticle,
						  BondType bondForce,
						  const int * __restrict__ id2index){
      extern __shared__ char shMem[];
      
      real4 *forceTotal = (real4*) shMem; //Each thread has a force

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
      
            
      real4 f = make_real4(real(0.0));          


      //__syncthreads();    
      for(int b = threadIdx.x; b<nbonds; b += blockDim.x){
	
	//Read bond info
	auto bond = bondList[b];	
    
	//Compute force
	const real3 r12 =  posi - bond.pos;
      
	const real fmod = bondForce.force(p,-1, r12, bondList[b].bond_info);                  

	f += make_real4(fmod*r12);
      }

      /*The first thread sums all the contributions*/
      forceTotal[threadIdx.x] = f;
      __syncthreads();
      //TODO Implement a warp reduction
      if(threadIdx.x==0){
	real4 ft = make_real4(0.0f);
	for(int i=0; i<blockDim.x; i++){
	  ft += forceTotal[i];
	}
	/*Write to global memory*/
	force[id2index[p]] += ft;
      }

    }
    

    
 
  }

  /*Perform an integration step*/
  template<class BondType>
  void BondedForces<BondType>::sumForce(cudaStream_t st){

    if(nbonds>0){
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto force = pd->getForce(access::location::gpu, access::mode::readwrite);

      auto id2index = pd->getIdOrderedIndices(access::location::gpu);
      
      auto d_bondStart = thrust::raw_pointer_cast(bondStart.data());
      auto d_nbondsPerParticle = thrust::raw_pointer_cast(nbondsPerParticle.data());
      uint Nparticles_with_bonds = bondStart.size();
      BondedForces_ns::computeBondedForces<Bond, BondType>
	<<<
	Nparticles_with_bonds,
	TPP,
	TPP*sizeof(real4),//+2*sizeof(int)+sizeof(real)+sizeof(Bond*),
	st>>>(
	      force.raw(), pos.raw(),
	      d_bondStart, d_nbondsPerParticle,
	      bondForce,
	      id2index);
    }
    if(nbondsFP>0){
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto force = pd->getForce(access::location::gpu, access::mode::readwrite);

      auto id2index = pd->getIdOrderedIndices(access::location::gpu);
      
      int numberParticlesWithBonds = bondStartFP.size();
      auto d_bondStart = thrust::raw_pointer_cast(bondStartFP.data());
      auto d_nbondsPerParticle = thrust::raw_pointer_cast(nbondsPerParticleFP.data());      
      BondedForces_ns::computeBondedForcesFixedPoint
	<<<numberParticlesWithBonds,	TPP, TPP*sizeof(real4), st>>>(
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
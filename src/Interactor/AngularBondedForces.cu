/*Raul P. Pelaez 2019. Angular bonds (bonds between three particles).

 */

#include"AngularBondedForces.cuh"
#include"global/defines.h"
#include<iostream>
#include<fstream>
#include<vector>
#include<set>
#include"utils/exception.h"
namespace uammd{

  template<class BondType>
  AngularBondedForces<BondType>::AngularBondedForces(shared_ptr<ParticleData> pd,
						     shared_ptr<System> sys,
						     Parameters par,
						     std::shared_ptr<BondType> bondType_in):
    Interactor(pd, sys, "AngularBondedForces"),
    TPP(32),
    bondType(bondType_in){
    this->setDelegate(bondType);
    auto bondProcessor = readBondFile(par.file);
    generateBondList(bondProcessor);
    bondProcessor.checkDuplicatedBonds();
  }

  template<class BondType>
  typename AngularBondedForces<BondType>::BondProcessor AngularBondedForces<BondType>::readBondFile(std::string bondFile){
    int numberParticles = pg->getNumberParticles();
    BondProcessor bondProcessor(numberParticles);
    BondReader bondReader(bondFile);
    this->nbonds = bondReader.getNumberBonds();
    bondProcessor.hintNumberBonds(nbonds);
    if(nbonds > 0){
      for(int b = 0; b < nbonds; b++){
	Bond bond;
	try{
	  bond = bondReader.readNextBond<Bond, BondType>();
	}
	catch(...){
	  sys->log<System::ERROR>("[AngularBondedForces] ERROR! Bond file ended too soon! Expected %d lines, found %d", nbonds, b);
	  throw;
	}
        bondProcessor.registerBond(bond);
      }
    }
    sys->log<System::MESSAGE>("[AngularBondedForces] Detected: %d bonds", nbonds);
    return std::move(bondProcessor);
  }

  template<class BondType>
  void AngularBondedForces<BondType>::generateBondList(const BondProcessor &bondProcessor){
    auto h_particlesWithBonds = bondProcessor.getParticlesWithBonds();
    sys->log<System::DEBUG>("[AngularBondedForces] Generating list");
    sys->log<System::MESSAGE>("[AngularBondedForces] %d particles are involved in at least one bond.", h_particlesWithBonds.size());
    const int NparticleswithBonds = h_particlesWithBonds.size();
    std::vector<int> h_bondStart, h_bondEnd;
    h_bondStart.resize(NparticleswithBonds, 0xffFFffFF);
    h_bondEnd.resize(NparticleswithBonds, 0);
    thrust::host_vector<Bond> bondListCPU(numberParticlesPerBond*nbonds);
    std::vector<Bond> blst;
    fori(0, NparticleswithBonds){
      const int index = h_particlesWithBonds[i];
      blst = bondProcessor.getBondListOfParticle(index);
      const int nbondsi = blst.size();
      int offset = (i>0)?h_bondEnd[i-1]:0;
      forj(0,nbondsi){
	bondListCPU[offset+j] = blst[j];
      }
      h_bondEnd[i] = offset + nbondsi;
      h_bondStart[i] = offset;
    }
    sys->log<System::DEBUG>("[AngularBondedForces] Uploading list");
    this->bondList = bondListCPU;
    this->bondStart = h_bondStart;
    this->bondEnd = h_bondEnd;
    this->particlesWithBonds = h_particlesWithBonds;
  }

  namespace Bonded_ns{
    template<class Bond, class BondType>
    __global__ void computeAngularBondedForce(real4* __restrict__ force,
					      const real4* __restrict__ pos,
					      const int* __restrict__ bondStart,
					      const int* __restrict__ bondEnd,
					      const int* __restrict__ particlesWithBonds,
					      const Bond* __restrict__ bondList,
					      const int * __restrict__ id2index,
					      BondType bondType){
      extern __shared__ real3 forceTotal[];
      const int tid = blockIdx.x;
      const int id_i = particlesWithBonds[tid];
      const int index = id2index[id_i];
      const real3 posp = make_real3(pos[index]);
      const int first = bondStart[tid];
      const int last = bondEnd[tid];
      real3 f = real3();
      for(int b = first + threadIdx.x; b < last; b += blockDim.x){
	auto bond = bondList[b];
	const int i = id2index[bond.i];
	const int j = id2index[bond.j];
	const int k = id2index[bond.k];
	real3 posi = (index==i)?posp:make_real3(pos[i]);
	real3 posj = (index==j)?posp:make_real3(pos[j]);
	real3 posk = (index==k)?posp:make_real3(pos[k]);
	f += bondType.force(i,j,k,
			    index,
			    posi, posj, posk,
			    bond.bond_info);
      }
      forceTotal[threadIdx.x] = f;
      __syncthreads();
      //TODO Implement a warp reduction
      if(threadIdx.x==0){
	real3 ft = make_real3(real(0.0));
	for(int i=0; i<blockDim.x; i++){
	  ft += forceTotal[i];
	}
	force[index] += make_real4(ft);
      }
    }
  }

  template<class BondType>
  void AngularBondedForces<BondType>::sumForce(cudaStream_t st){
    if(nbonds>0){
      int Nparticles_with_bonds = particlesWithBonds.size();
      auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto d_bondStart = thrust::raw_pointer_cast(bondStart.data());
      auto d_bondEnd = thrust::raw_pointer_cast(bondEnd.data());
      auto d_particlesWithBonds = thrust::raw_pointer_cast(particlesWithBonds.data());
      auto d_bondList = thrust::raw_pointer_cast(bondList.data());
      auto id2index = pd->getIdOrderedIndices(access::location::gpu);
      Bonded_ns::computeAngularBondedForce<<<Nparticles_with_bonds,
	TPP,
	TPP*sizeof(real3)>>>(force.raw(), pos.raw(),
			     d_bondStart,
			     d_bondEnd,
			     d_particlesWithBonds,
			     d_bondList,
			     id2index,
			     *bondType);
    }

  }


  namespace Bonded_ns{
    template<class Bond, class BondType>
    __global__ void computeAngularBondedEnergy(real* __restrict__ energy,
					      const real4* __restrict__ pos,
					      const int* __restrict__ bondStart,
					      const int* __restrict__ bondEnd,
					      const int* __restrict__ particlesWithBonds,
					      const Bond* __restrict__ bondList,
					      const int * __restrict__ id2index,
					      BondType bondType){
      extern __shared__ real energyTotal[];
      const int tid = blockIdx.x;
      const int id_i = particlesWithBonds[tid];
      const int index = id2index[id_i];
      const real3 posp = make_real3(pos[index]);
      const int first = bondStart[tid];
      const int last = bondEnd[tid];
      real f = real();
      for(int b = first + threadIdx.x; b < last; b += blockDim.x){
	auto bond = bondList[b];
	const int i = id2index[bond.i];
	const int j = id2index[bond.j];
	const int k = id2index[bond.k];
	real3 posi = (index==i)?posp:make_real3(pos[i]);
	real3 posj = (index==j)?posp:make_real3(pos[j]);
	real3 posk = (index==k)?posp:make_real3(pos[k]);
	f += bondType.energy(i,j,k,
			     index,
			     posi, posj, posk,
			     bond.bond_info);
      }
      energyTotal[threadIdx.x] = f;
      __syncthreads();
      //TODO Implement a warp reduction
      if(threadIdx.x==0){
	real ft = real(0.0);
	for(int i=0; i<blockDim.x; i++){
	  ft += energyTotal[i];
	}
	energy[index] += ft;
      }
    }
  }

  template<class BondType>
  real AngularBondedForces<BondType>::sumEnergy(){
    if(nbonds>0){
      int Nparticles_with_bonds = particlesWithBonds.size();
      auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto d_bondStart = thrust::raw_pointer_cast(bondStart.data());
      auto d_bondEnd = thrust::raw_pointer_cast(bondEnd.data());
      auto d_particlesWithBonds = thrust::raw_pointer_cast(particlesWithBonds.data());
      auto d_bondList = thrust::raw_pointer_cast(bondList.data());
      auto id2index = pd->getIdOrderedIndices(access::location::gpu);
      Bonded_ns::computeAngularBondedEnergy<<<Nparticles_with_bonds,
	TPP,
	TPP*sizeof(real)>>>(energy.begin(), pos.begin(),
			     d_bondStart,
			     d_bondEnd,
			     d_particlesWithBonds,
			     d_bondList,
			     id2index,
			     *bondType);
    }

    return 0;
  }

  // real AngularBondedForces::sumVirial(){
  //   return 0;
  // }

}

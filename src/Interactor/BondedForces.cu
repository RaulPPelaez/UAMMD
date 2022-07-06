/*
  Raul P. Pelaez 2016-2021. Bonded pair forces Interactor implementation. i.e two body springs

  See BondedForces.cuh for more info.

TODO:
100 - Implement a warp reduction in computeForces
*/

#include"BondedForces.cuh"
#include<third_party/uammd_cub.cuh>
#include <stdexcept>
#include<thrust/sort.h>
#include<thrust/reduce.h>
#include<algorithm>
#include<cstdint>
#include<vector>
#include<set>
#include<fstream>
namespace uammd{
  template<class BondType, int particlesPerBond>
  BondedForces<BondType, particlesPerBond>::BondedForces(shared_ptr<ParticleData> pd,
							 Parameters par,
							 std::shared_ptr<BondType> bondCompute):
    Interactor(pd, "BondedForces/" + type_name<BondType>()),
    bondCompute(bondCompute), TPP(64){
    //BondedForces does not care about any parameter update, but the BondType might.
    this->setDelegate(this->bondCompute);
    System::log<System::MESSAGE>("[BondedForces] Initialized");
    System::log<System::MESSAGE>("[BondedForces] Using: %s", type_name<BondType>().c_str());
    readBonds(par.file);
  }

  namespace bondedforces_ns{
    template<class Bond>
    class BondProcessor{
      std::vector<std::vector<int>> isInBonds;
      std::vector<Bond> bondList;
      std::set<int> particlesWithBonds;

      void registerParticleInBond(int particleIndex, int b){
	if(particleIndex >= isInBonds.size()){
	  isInBonds.resize(particleIndex + 1);
	}
	isInBonds[particleIndex].push_back(b);
	particlesWithBonds.insert(particleIndex);
      }

    public:

      void hintNumberBonds(int nbonds){
	bondList.reserve(nbonds);
      }

      void registerBond(Bond b, int particlesPerBond){
	int bondIndex = bondList.size();
	bondList.push_back(b);
	for(int i = 0; i<particlesPerBond; i++){
	  int id = b.ids[i];
	  if(id>=0){ //Fixed point bonds have negative index
	    registerParticleInBond(b.ids[i], bondIndex);
	  }
	}
      }

      std::vector<int> getParticlesWithBonds() const{
	return {particlesWithBonds.begin(), particlesWithBonds.end()};
      }

      std::vector<Bond> getBondListOfParticle(int index) const{
	std::vector<Bond> blst;
	blst.resize(isInBonds[index].size());
	fori(0, blst.size()){
	  blst[i] = bondList[isInBonds[index][i]];
	}
	return blst;
      }
    };

    template<class Bond, class BondType>
    Bond readNextBond(std::ifstream &in, int currentBonds, int nbonds, int particlesPerBond){
      Bond bond;
      for(int i = 0; i< particlesPerBond; i++){
	if(!(in>>bond.ids[i])){
	  System::log<System::EXCEPTION>("[BondedForces] ERROR! Bond file ended too soon! Expected %d lines, found %d",
					 nbonds, currentBonds);
	  throw std::ios_base::failure("File unreadable");
	}
      }
      bond.bond_info = BondType::readBond(in);
      return bond;
    }

    template<class Bond, class BondType>
    Bond readNextBondFixedPoint(std::ifstream &in, int currentBonds, int nbonds, real3 &pos){
      Bond bond;
      if(!(in>>bond.ids[0])){
	System::log<System::EXCEPTION>("[BondedForces] ERROR! Bond file ended too soon! Expected %d lines, found %d",
				       nbonds, currentBonds);
	throw std::ios_base::failure("File unreadable");
      }
      bond.ids[1] = -(currentBonds+1);
      in>>pos;
      bond.bond_info = BondType::readBond(in);
      return bond;
    }

    template<class Bond>
    auto buildBondList(const BondProcessor<Bond> & processor, int particlesPerBond, int nbonds){
      System::log<System::DEBUG>("[BondedForces] Generating list");
      auto h_particlesWithBonds = processor.getParticlesWithBonds();
      const int NparticleswithBonds = h_particlesWithBonds.size();
      std::vector<int> h_bondStart, h_bondEnd;
      h_bondStart.resize(NparticleswithBonds, 0xffFFffFF);
      h_bondEnd.resize(NparticleswithBonds, 0);
      std::vector<Bond> bondListCPU(particlesPerBond*nbonds);
      std::vector<Bond> blst;
      //Create a compact list of bonds for each particle that has at least one bond.
      //The bond list of particle with id=particlesWithBonds[i] starts at bondStart[i] and ends at bondEnd[i].
      //So the first bond is bondList[bondStart[i]]
      fori(0, NparticleswithBonds){
	const int index = h_particlesWithBonds[i];
	blst = processor.getBondListOfParticle(index);
	const int nbondsi = blst.size();
	int offset = (i>0)?h_bondEnd[i-1]:0;
	forj(0, nbondsi){
	  bondListCPU[offset+j] = blst[j];
	}
	h_bondEnd[i] = offset + nbondsi;
	h_bondStart[i] = offset;
      }
      return std::make_tuple(bondListCPU, h_bondStart, h_bondEnd);
    }

  }



  template<class BondType, int particlesPerBond>
  void BondedForces<BondType, particlesPerBond>::readBonds(std::string fileName){
    const int numberParticles = pg->getNumberParticles();
    using namespace bondedforces_ns;
    std::ifstream in(fileName);
    if(not in){
      throw std::runtime_error("[BondedForces] File " + fileName + " cannot be opened.");
    }
    in>>this->nbonds;
    BondProcessor<Bond> processor;
    if(nbonds > 0){
      System::log<System::MESSAGE>("[BondedForces] Detected: %d bonds", nbonds);
      processor.hintNumberBonds(nbonds); //This makes the processor reserve the space for the bond list in advance
      for(int b = 0; b < nbonds; b++){
	auto bond = readNextBond<Bond, BondType>(in, b, nbonds, particlesPerBond);
        processor.registerBond(bond, particlesPerBond);
      }
    }
    //Fixed point bonds are treated as pair bonds were one of the particles is attached to a point in space
    if(particlesPerBond == 2){
      int nbondsFP = 0;
      in>>nbondsFP;
      if(nbondsFP){
	nbonds += nbondsFP;
	System::log<System::MESSAGE>("[BondedForces] Detected: %d fixed point bonds", nbondsFP);
	processor.hintNumberBonds(nbonds); //This makes the processor reserve the space for the bond list in advance
	this->fixedPointPositions.resize(nbondsFP+1);
	for(int b = 0; b < nbondsFP; b++){
	  real3 pos;
	  auto bond = readNextBondFixedPoint<Bond, BondType>(in, b, nbondsFP, pos);
	  fixedPointPositions[b+1] = make_real4(pos);
	  processor.registerBond(bond, particlesPerBond);
	}
      }
    }
    std::tie(this->bondList, this->bondStart, this->bondEnd) = buildBondList(processor, particlesPerBond, nbonds);
    this->particlesWithBonds = processor.getParticlesWithBonds();
    System::log<System::MESSAGE>("[BondedForces] %d particles are involved in at least one bond.",
				 particlesWithBonds.size());
  }

  namespace BondedForces_ns{

    template<int particlesPerBond, bool fpb, class Bond, class BondCompute>
    __global__ void computeBondedThreadPerParticle(real4* force,
						   real*  energy,
						   real*  virial,
						   Interactor::Computables comp,
						   const real4* pos,
						   const Bond* bondList,
						   const int* bondStart,
						   const int *bondEnd,
						   const int *particlesWithBonds,
						   const real4* fixedPointPositions,
						   BondCompute bondCompute,
						   const int * id2index, int N){
      int tid = blockIdx.x*blockDim.x + threadIdx.x;
      if(tid>=N) return;
      const int id_i = particlesWithBonds[tid];
      const int index = id2index[id_i]; //Current index of the particle
      const real3 posi = make_real3(pos[index]);
      ComputeType ct;
      const int first = bondStart[tid];
      const int last = bondEnd[tid];
      #pragma unroll 3
      for(int b = first; b<last; b++){
	const auto bond = bondList[b];
	int indexes[particlesPerBond]; //Current indexes of the particles involved in the bond
	real3 positions[particlesPerBond]; //Positions of the particles involved in the bond
        #pragma unroll particlesPerBond
	for(int i = 0; i<particlesPerBond; i++){
	  const int id_j = bond.ids[i];
	  //Fixed point bonds are treated as a special case of two particles, with one of them having a negative index
	  const int index_j = (fpb and id_j<0)?-1:id2index[id_j];
	  indexes[i] = index_j;
	  positions[i] = make_real3((fpb and id_j<0)?fixedPointPositions[-id_j]:pos[index_j]);
	}
	const auto ctt = bondCompute.compute(index, indexes, positions, comp, bond.bond_info);
	ct.force += ctt.force;
	ct.virial += ctt.virial;
	ct.energy += ctt.energy;
      }
      if(comp.force)  force[index]  += make_real4(ct.force);
      if(comp.energy) energy[index] += ct.energy;
      if(comp.virial) virial[index] += ct.virial;
    }

    template<int particlesPerBond, bool fpb, class ...T>
    void dispatchComputeBonded(int Nparticles_with_bonds, int TPB,  cudaStream_t st, T...args){
      int Nthreads= 128;
      int Nblocks=Nparticles_with_bonds/Nthreads + ((Nparticles_with_bonds%Nthreads)?1:0);
      computeBondedThreadPerParticle<particlesPerBond, fpb><<<Nblocks, Nthreads, 0, st>>>(args...);
      //A block per particle might be benefitial, but lets leave it off for now
      // if(TPB<=32 or Nparticles_with_bonds < 5000){ //Empirical magic numbers, could probably be chosen better
      // }
      // else if(TPB>=128){
      // 	//This is due to cub having blocksize as template parameter, I hate it
      // 	//computeBondedBlockPerParticle<128><<<Nparticles_with_bonds, 128, 0, st>>>(args...);
      // }
      // else{
      // 	//computeBondedBlockPerParticle<64><<<Nparticles_with_bonds, 64, 0, st>>>(args...);
      // }
    }
  }

  //This function chooses which version of the kernel computeBondedForces to use as a function of the number of bonds per particle
  template<class BondType, int particlesPerBond>
  void BondedForces<BondType, particlesPerBond>::callComputeBonded(real4* f, real*e, real*v, cudaStream_t st){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto id2index = pd->getIdOrderedIndices(access::location::gpu);
    auto d_bondStart = thrust::raw_pointer_cast(bondStart.data());
    auto d_bondEnd = thrust::raw_pointer_cast(bondEnd.data());
    auto d_bondList = thrust::raw_pointer_cast(bondList.data());
    auto d_particlesWithBonds = thrust::raw_pointer_cast(particlesWithBonds.data());
    auto d_fixedPointPositions = thrust::raw_pointer_cast(fixedPointPositions.data());
    uint Nparticles_with_bonds = bondStart.size();
    Interactor::Computables comp{.force=f!=nullptr, .energy=e!=nullptr, .virial=v!=nullptr};
    if(d_fixedPointPositions){
      BondedForces_ns::dispatchComputeBonded<particlesPerBond, true>(Nparticles_with_bonds, TPP, st,
								     f,e,v, comp, pos.raw(),
								     d_bondList, d_bondStart, d_bondEnd, d_particlesWithBonds,
								     d_fixedPointPositions,
								     *bondCompute, id2index, Nparticles_with_bonds);
    }
    else{
      BondedForces_ns::dispatchComputeBonded<particlesPerBond, false>(Nparticles_with_bonds, TPP, st,
								      f,e,v, comp, pos.raw(),
								      d_bondList, d_bondStart, d_bondEnd, d_particlesWithBonds,
								      d_fixedPointPositions,
								      *bondCompute, id2index, Nparticles_with_bonds);
    }
  }

  template<class BondType, int particlesPerBond>
  void BondedForces<BondType, particlesPerBond>::sum(Computables comp, cudaStream_t st){
    auto force  = comp.force?pd->getForce(access::gpu, access::readwrite).begin():nullptr;
    auto energy = comp.energy?pd->getEnergy(access::gpu, access::readwrite).begin():nullptr;
    auto virial = comp.virial?pd->getVirial(access::gpu, access::readwrite).begin():nullptr;
    if(nbonds>0){
      System::log<System::DEBUG3>("[BondedForces] Computing Particle-Particle...");
      this->callComputeBonded(force, energy, virial, st);
    }
  }
}

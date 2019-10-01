/*Raul P. Pelaez 2017


 */



#include"AngularBondedForces.cuh"
#include"global/defines.h"
#include<iostream>
#include<fstream>
#include<vector>
#include<set>

namespace uammd{


  template<class BondType>
  AngularBondedForces<BondType>::AngularBondedForces(shared_ptr<ParticleData> pd,
						     shared_ptr<System> sys,
						     Parameters par,
						     BondType bondType_in):
    Interactor(pd,
	       sys,
	       "AngularBondedForces"),
    TPP(32),
    bondType(bondType_in){

    this->setDelegate(&bondType);
    int numberParticles = pg->getNumberParticles();
    sys->log<System::MESSAGE>("[AngularBondedForces] Initialized");

    nbonds = 0;
    /*Read the bond list from the file*/
    std::ifstream in(par.readFile);
    //Has a list of the bonds each particle is in
    std::vector<std::vector<int>> isInBonds(numberParticles);
    if(!in)
      sys->log<System::CRITICAL>("[AngularBondedForces] File %s cannot be opened.", par.readFile.c_str());

    in>>nbonds;
    std::vector<Bond> blst(nbonds); //Temporal storage for the bonds in the file


    sys->log<System::MESSAGE>("[AngularBondedForces] Detected: %d particle-particle-particle bonds", nbonds);

    if(nbonds>0){
      std::set<int> pwb; //Particles with bonds
      for(int b=0; b<nbonds; b++){
	int i, j, k;
	if(!(in>>i>>j>>k))
	  sys->log<System::CRITICAL>("[AngularBondedForces] ERROR! Bond file ended too soon! Expected %d lines, found %d", nbonds, b);

	isInBonds[i].push_back(b);
	isInBonds[j].push_back(b);
	isInBonds[k].push_back(b);

	blst[b].i = i;
	blst[b].j = j;
	blst[b].k = k;

	blst[b].bond_info = BondType::readBond(in);

	pwb.insert(i);
	pwb.insert(j);
	pwb.insert(k);
      }
      particlesWithBonds.assign(pwb.begin(), pwb.end());
    }


    const int NparticleswithBonds = particlesWithBonds.size();

    bondStart.resize(NparticleswithBonds, 0xffFFffFF);
    bondEnd.resize(NparticleswithBonds, 0);

    thrust::host_vector<Bond> bondListCPU(3*nbonds);

    //Fill bondList, bondStart and bondEnd
    //BondList has the following format:
    //[all bonds involving particle 0, all bonds involving particle 1...]
    //So it has size 3*nbonds
    fori(0, NparticleswithBonds){

      const int index = particlesWithBonds[i];
      const int nbondsi = isInBonds[index].size();

      int offset;
      if(i>0) offset = bondEnd[i-1];
      else    offset = 0;
      forj(0,nbondsi){
	bondListCPU[offset+j] = blst[isInBonds[index][j]];
      }
      bondEnd[i] = offset+nbondsi;
      bondStart[i] = offset;
    }


    nbonds *= 3; //We store all the bonds in which every particle is involved, per particle.

    //Upload bondList to GPU
    bondList = bondListCPU;
    sys->log<System::MESSAGE>("[AngularBondedForces] %d particles are involved in at least one bond.",particlesWithBonds.size());

    struct AngularBondCompareLessThan{
      bool operator()(const Bond& lhs, const Bond &rhs) const{
	if((lhs.i < rhs.i and lhs.j < rhs.j and lhs.k < rhs.k)
	   or (lhs.i < rhs.k and lhs.j < rhs.j and lhs.k < rhs.i))
	  return true;
	else return false;
      }
    };


    {
      std::set<Bond, AngularBondCompareLessThan> checkDuplicates;

      fori(0, blst.size()){
	if(!checkDuplicates.insert(blst[i]).second)
	  sys->log<System::WARNING>("[AngularBondedForces] Bond %d %d %d with index %d is duplicated!", blst[i].i, blst[i].j, blst[i].k, i);
      }
    }
  }




  namespace Bonded_ns{
    //Custom kernel to compute and sum the force in a three particle angle spring
    /*
      Computes the potential: V(theta) = 2.0 K(sin(theta/2)-sin(theta_0/2))^2
      F(\vec{ri}) = d(V(theta))/d(cos(theta))·d(cos(theta))/d(\vec{ri})
    */
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
      //A block per particle
      const int tid = blockIdx.x;

      //Id of the first particle with bonds
      const int id_i = particlesWithBonds[tid];

      //Current index of my particle in the global arrays
      const int index = id2index[id_i];
      const real3 posp = make_real3(pos[index]);

      const int first = bondStart[tid];
      const int last = bondEnd[tid];

      real3 f = make_real3(real(0.0));


      real3 posi,posj, posk; //The bond particles
      //Go through my bonds
      for(int b = first+threadIdx.x; b<last; b+=blockDim.x){
	//Recover bond info
	auto bond = bondList[b];
	const int i = id2index[bond.i];
	const int j = id2index[bond.j];
	const int k = id2index[bond.k];

	//TODO Texture memory target
	//Store the positions of the three particles
	//We already got one of them, p
	//Differentiate between the three particles in the bond
	if(index==i){
	  posi = posp;
	  posj = make_real3(pos[j]);
	  posk = make_real3(pos[k]);
	}
	else if(index==j){
	  posi = make_real3(pos[i]);
	  posj = posp;
	  posk = make_real3(pos[k]);
	}
	else{
	  posi = make_real3(pos[i]);
	  posj = make_real3(pos[j]);
	  posk = posp;
	}

	f += bondType.force(i,j,k,
			    index,
			    posi, posj, posk,
			    bond.bond_info);
      }

      //The fisrt thread sums all the contributions
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
			     bondType);

    }

  }

  template<class BondType>
  real AngularBondedForces<BondType>::sumEnergy(){
    return 0;
  }

  // real AngularBondedForces::sumVirial(){
  //   return 0;
  // }

}

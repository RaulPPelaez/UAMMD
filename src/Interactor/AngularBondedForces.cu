/*Raul P. Pelaez 2017


 */

/********************************THREE BONDED FORCES**********************************/

#include"AngularBondedForces.cuh"
#include"global/defines.h"
#include<iostream>
#include<fstream>
#include<vector>

namespace uammd{


  AngularBondedForces::~AngularBondedForces(){}


  AngularBondedForces::AngularBondedForces(shared_ptr<ParticleData> pd,
					   shared_ptr<System> sys,
					   Parameters par):
    Interactor(pd,
	       sys,
	       "AngularBondedForces"),
    TPP(64),
    box(par.box){
    
    int numberParticles = pg->getNumberParticles();
    sys->log<System::MESSAGE>("[AngularBondedForces] Initialized");

    nbonds = 0;
    /*Read the bond list from the file*/
    std::ifstream in(par.readFile);
    std::vector<std::vector<int>> isInBonds(numberParticles);
    in>>nbonds;
    std::vector<ThreeBond> blst(nbonds); //Temporal storage for the bonds in the file

  
    sys->log<System::MESSAGE>("[AngularBondedForces] Detected: %d particle-particle-particle bonds", nbonds);
    
    if(nbonds>0){
    
      for(int b=0; b<nbonds; b++){
	int i, j, k;
	in>>i>>j>>k;
	
	isInBonds[i].push_back(b);
	isInBonds[j].push_back(b);
	isInBonds[k].push_back(b);      
      
	blst[b].i = i;
	blst[b].j = j;
	blst[b].k = k;
      
	in>>blst[b].kspring>>blst[b].ang;
	
      }
    }
  
  
    bondList.resize(nbonds*3);//Allocate 3*nbonds
    bondStart.resize(numberParticles, 0xffFFffFF);
    bondEnd.resize(numberParticles, 0);

  
    fori(0, numberParticles){
      int nbondsi;
      nbondsi = isInBonds[i].size();
      if(nbondsi==0) continue;
    
      int offset;
      if(i>0)
	offset = bondEnd[i-1];
      else
	offset = 0;
    
      forj(0,nbondsi){
	bondList[offset+j] = blst[isInBonds[i][j]];
      }
      bondEnd[i] = offset+nbondsi;
      bondStart[i] = offset;
    }

    std::vector<int> pwb; //Particles with bonds
    fori(0, numberParticles){
      if(bondStart[i]!=0xffFFffFF){
	pwb.push_back(i);
      }
    }

    bondParticleIndex.assign(pwb.begin(), pwb.end());
  
    nbonds *= 3; //We store all the bonds in which every particle is involved, per particle.
   
    sys->log<System::MESSAGE>("[AngularBondedForces] %d particles are involved in at least one bond.",pwb.size());

  }




  namespace Bonded_ns{
    //Custom kernel to compute and sum the force in a three particle angle spring
    /*
      Computes the potential: V(theta) = 0.5 K(theta-theta_0)^2
      F(\vec{ri}) = d(V(theta))/d(cos(theta))·d(cos(theta))/d(\vec{ri})
    */
    __global__ void computeThreeBondedForce(real4* __restrict__ force,
					    const real4* __restrict__ pos,
					    const int* __restrict__ bondStart,
					    const int* __restrict__ bondEnd,
					    const int* __restrict__ bondedParticleIndex,
					    const AngularBondedForces::ThreeBond* __restrict__ bondList,
					    const int * __restrict__ id2index,
					    Box box){
      extern __shared__ real3 forceTotal[];
      /*A block per particle, as in computeBondedForcesD*/
      int p = id2index[bondedParticleIndex[blockIdx.x]];

      real3 posp = make_real3(pos[p]);
  
      int first = bondStart[p];
      int last = bondEnd[p];
   
      real3 f = make_real3(real(0.0));

      int i,j,k;             //The bond indices
      real3 posi,posj, posk; //The bond particles
      real kspring, ang0; //The bond info

      /*         i -------- j -------- k*/
      /*             rij->      <-rkj  */
    
      real3 rij, rkj; //rij = ri - rj
  
      real invsqrij, invsqrkj; //1/|rij|
      real rij2, rkj2;  //|rij|^2

    
      real a2; 
      real cijk, sijk;
      real a, a11, a12, a22;
      real ampli;

      /*Go through my bonds*/
      for(int b = first+threadIdx.x; b<last; b+=blockDim.x){
	/*Recover bond info*/
	auto bond = bondList[b];
	i = id2index[bond.i];
	j = id2index[bond.j];
	k = id2index[bond.k];

	kspring = bond.kspring;	
	ang0 = bond.ang;

      
	//TODO Texture memory target
	/*Store the positions of the three particles*/
	/*We already got one of them, p*/
	/*Differentiate between the three particles in the bond*/
	if(p==i){
	  posi = posp;
	  posj = make_real3(pos[j]);
	  posk = make_real3(pos[k]);
	}
	else if(p==j){
	  posi = make_real3(pos[i]);
	  posj = posp;
	  posk = make_real3(pos[k]);
	}
	else{
	  posi = make_real3(pos[i]);
	  posj = make_real3(pos[j]);
	  posk = posp;
	}

	/*Compute distances and vectors*/
	/***rij***/
	rij =  box.apply_pbc(posi-posj);
	rij2 = dot(rij, rij);
	invsqrij = rsqrt(rij2);
	/***rkj***/
	rkj =  box.apply_pbc(posk-posj);
	rkj2 = dot(rkj, rkj);
	invsqrkj = rsqrtf(rkj2);
	/********/
      
	a2 = invsqrij * invsqrkj;
	cijk = dot(rij, rkj)*a2; //cijk = cos (theta) = rij*rkj / mod(rij)*mod(rkj)

	/*Cos must stay in range*/
	if(cijk>real(1.0)) cijk = real(1.0);
	else if (cijk<real(-1.0)) cijk = -real(1.0);


	//Approximation for small angle displacements	
	//sijk = sqrt(real(1.0)-cijk*cijk); //sijk = sin(theta) = sqrt(1-cos(theta)^2)
	//sijk cant be zero to avoid division by zero
	//if(sijk<real(0.000001)) sijk = real(0.000001);
	//ampli = -kspring * (acosf(cijk) - ang0)/sijk; //The force amplitude -k·(theta-theta_0)

	
	if(ang0 == real(0.0) && false){
	  //TODO replace rij for rji so ang0=0 means straight and this can apply
	  //When ang0=pi means stragiht it is difficult to check if ang0 is pi
	  ampli = -kspring;
	}
	else{
	  const real theta = acosf(cijk);
	  const real sinthetao2 = sinf(real(0.5)*theta);
	  ampli = -kspring*(sintheto2- sinf(ang0*real(0.5)))/sinthetao2;
	}
	
	//ampli = -kang*(-sijk*cos(ang0)+cijk*sin(ang0))+ang0; //k(1-cos(ang-ang0))
	
	

	//Magical trigonometric relations to infere the direction of the force

	a11 = ampli*cijk/rij2;
	a12 = -ampli*a2;
	a22 = ampli*cijk/rkj2;
      
	/*Sum according to my position in the bond*/
	// i ----- j ------ k
	if(p==i){
	  f += make_real3(a11*rij + a12*rkj); //Angular spring	
	}
	else if(p==j){
	  //Angular spring
	  f -= make_real3(a11*rij + a12*rkj + a22*rkj + a12*rij);	
	}
	else if(p==k){
	  //Angular spring
	  f += make_real3(a22*rkj + a12*rij);
	}
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
	force[p] += make_real4(ft);
      }

    }



  }
  void AngularBondedForces::sumForce(cudaStream_t st){
    if(nbonds>0){
      int Nparticles_with_bonds = bondParticleIndex.size();
      
      auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto d_bondStart = thrust::raw_pointer_cast(bondStart.data());
      auto d_bondEnd = thrust::raw_pointer_cast(bondEnd.data());
      auto d_bondParticleIndex = thrust::raw_pointer_cast(bondParticleIndex.data());
      auto d_bondList = thrust::raw_pointer_cast(bondList.data());

      auto id2index = pd->getIdOrderedIndices(access::location::gpu);
      
      Bonded_ns::computeThreeBondedForce<<<Nparticles_with_bonds,
	TPP,
	TPP*sizeof(real3)>>>(force.raw(), pos.raw(),
			     d_bondStart,
			     d_bondEnd,
			     d_bondParticleIndex,
			     d_bondList,
			     id2index,
			     box);

    }

  }


  real AngularBondedForces::sumEnergy(){
    return 0;
  }

  // real AngularBondedForces::sumVirial(){
  //   return 0;
  // }

}

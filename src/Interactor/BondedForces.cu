/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs

  See BondedForces.cuh for more info.

*/


/*Constructors*/
template<class BondType>
BondedForces<BondType>::BondedForces(const char * readFile):
BondedForces<BondType>(readFile, BondType(), gcnf.L, gcnf.N){}

template<class BondType>
BondedForces<BondType>::BondedForces(const char * readFile, BondType bondForce):
  BondedForces<BondType>(readFile, bondForce, gcnf.L, gcnf.N){}

template<class BondType>
BondedForces<BondType>::BondedForces(const char * readFile, BondType bondForce, real3 L, int N):  
  Interactor(64, L, N), bondForce(bondForce), TPP(64){

  name = "BondedForces";
    
  Nthreads = TPP<N?TPP:N;
  Nblocks = N/Nthreads + ((N%Nthreads!=0)?1:0);
  cerr<<"Initializing Bonded Forces..."<<endl;


  nbonds = nbondsFP = 0;
  /*If some bond type number is zero, the loop will simply not be entered, and no storage will be used*/
  /*Read the bond list from the file*/
  std::ifstream in(readFile);
  if(!in.good()){cerr<<"\tERROR: Bond file not found!!"<<endl; exit(1);}
  in>>nbonds;
  if(nbonds>0){
    bondList = Vector<Bond>(nbonds*2);//Allocate 2*nbonds, see init for explication
    fori(0, nbonds){
      in>>bondList[i].i>>bondList[i].j;
      bondList[i].bond_info = BondType::readBond(in);
    }
  }
  /*Fixed point bonds*/
  in>>nbondsFP;
  if(nbondsFP>0){
    bondListFP = Vector<BondFP>(nbondsFP);
    fori(0, nbondsFP){
      in>>bondListFP[i].i;
      in>>bondListFP[i].pos.x>>bondListFP[i].pos.y>>bondListFP[i].pos.z;
      bondListFP[i].bond_info = BondType::readBond(in);

    }
  }


  /*Upload and init GPU*/
  init();
  
  cerr<<"Bonded Forces\t\tDONE!!\n\n";
}


template<class BondType>
BondedForces<BondType>::~BondedForces(){}

//Criterion to sort bonds

template<class BondType>
bool bondComp(const BondType &a, const BondType &b){ return a.i<b.i;}

//Initialize variables and upload them to GPU, init CUDA
template<class BondType>
void BondedForces<BondType>::init(){
  /****************************************Pair bonds*********************************************/
  /*This algorithm is identical to the one used in PairForces to sort by cell*/
  /*First store all bonded pairs. That means i j and j i*/
  /*The first ones are readed given, the complementary have to be generated*/

  if(nbonds>0){
    fori(nbonds, 2*nbonds){
      bondList[i].i = bondList[i-nbonds].j;
      bondList[i].j = bondList[i-nbonds].i;
    
      bondList[i].bond_info = bondList[i-nbonds].bond_info;
    }
    /*Sort in the i index to construct bondStart and bondEnd*/
    std::sort(bondList.begin(), bondList.end(), bondComp<Bond>);
    
    /*Check for duplicates i.e the file contains the ij and ji bonds already*/
    // set<Bond,
    // 	bool(*)(const Bond& lhs, const Bond& rhs)> defbondlist(
    // 							       bondList.begin(), bondList.end(),
    // vector<Bond> defbondlist;    
    // fori(1,2*nbonds){
    //   if(bondList[i].i==bondList[i-1].i && bondList[i].j==bondList[i-1].j){
	
    //   }
    //   else{
    // 	defbondlist.push_back(bondList[i-1]);
    //   }

    // }
    // defbondlist.push_back(bondList[nbonds-1]);
    // bondList.assign(defbondlist.begin(), defbondlist.end());
    // /*Now sort the bondList by the first and second particle, i, j*/
    // std::sort(bondList.begin(), bondList.end(), bondCompj);
    // std::stable_sort(bondList.begin(), bondList.end(), bondComp);

    nbonds = bondList.size();
    /*We have a list of bonds ordered by its first particle, so; All the particles
      bonded with particle i=0, all particles "" i=1...*/

    /*We need additional arrays to know where in the list the bonds of particle i start
      and end*/
    /*Initially all bondStarts are 2^32-1, this value means no particles bonded*/
    bondStart = Vector<uint>(N); bondStart.fill_with(0xffFFffFF);
    bondEnd   = Vector<uint>(N); bondEnd.fill_with(0);

    /*Construct bondStart and bondEnd*/
    uint b, bprev = 0;
    for(uint i = 0; i<nbonds; i++){
      b = bondList[i].i; //Get my particle i
      if(i>0) bprev = bondList[i-1].i; //Get the previous's bond particle i

      /*If I am the first bond or my i is different than the previous bond
	I am the first bond of the particle*/
      if(i==0 || b !=bprev){
	bondStart[b] = i;
	/*And unless I am the first particle, I am also the last bond of the previous particle*/
	if(i>0)
	  bondEnd[bprev] = i;
      }
      /*Fix the last particle bondEnd*/
      if(i == nbonds-1) bondEnd[b] = i+1;
    }


    
    std::set<uint> particlesWithBonds;
    fori(0, bondList.size()){
      particlesWithBonds.insert(bondList[i].i);
    }

    bondParticleIndex.assign(particlesWithBonds.begin(), particlesWithBonds.end());

    /*Upload all to GPU*/
    bondParticleIndex.upload();
    bondList.upload();
    bondStart.upload();
    bondEnd.upload();
  }
  /************************************FixedPoint************************************************/
  if(nbondsFP>0){
    std::sort(bondListFP.data, bondListFP.data+nbondsFP, bondComp<BondFP>);
    bondStartFP = Vector<uint>(N); bondStartFP.fill_with(0xffffffff);
    bondEndFP   = Vector<uint>(N); bondEndFP.fill_with(0);

    /*Construct bondStart and bondEnd*/
    uint b, bprev = 0;
    for(uint i = 0; i<nbondsFP; i++){
      b = bondListFP[i].i; //Get my particle i
      if(i>0) bprev = bondListFP[i-1].i; //Get the previous's bond particle i

      /*If I am the first bond or my i is different than the previous bond
	I am the first bond of the particle*/
      if(i==0 || b !=bprev){
	bondStartFP[b] = i;
	/*And unless I am the first particle, I am also the last bond of the previous particle*/
	if(i>0)
	  bondEndFP[bprev] = i;
      }
      /*Fix the last particle bondEnd*/
      if(i == nbondsFP-1) bondEndFP[b] = i+1;
    }
  
      /*Upload all to GPU*/
      bondListFP.upload();
      bondStartFP.upload();
      bondEndFP.upload();
  }

  /***********************************************************************************************/
  cerr<<"\tDetected: "<<bondList.size()/2<<" particle-particle bonds and "<<bondListFP.size()<<" Fixed Point bonds"<<endl;
  cerr<<"\t"<<max(bondParticleIndex.size(), bondListFP.size())<<" particles have at least one bond"<<endl;
}



namespace Bonded_ns{
  
  template<class BondType>
  __global__ void computeBondedForces(real4* __restrict__ force, const real4* __restrict__ pos,
				      const uint* __restrict__ bondStart,
				      const uint* __restrict__ bondEnd,
				      const uint* __restrict__ bondedParticleIndex,
				      typename BondedForces<BondType>::Bond * __restrict__ bondList,
				      BondType bondForce){
    extern __shared__ real4 forceTotal[]; /*Each thread*/
    
    /*A block per particle*/
    /*Instead of launching a thread per particle and discarding those without any bond,
      I store an additional array of size N_particles_with_bonds that contains the indices
      of the particles that are involved in at least one bond. And only launch N_particles_with_bonds blocks*/
    /*Each thread in a block computes the force on particle p due to one (or several) bonds*/
    uint p = bondedParticleIndex[blockIdx.x]; //This block handles particle p
    real4 posi = pos[p]; //Get position

    /*First and last bond indices of p in bondList*/
    uint first = bondStart[p]; 
    uint last = bondEnd[p];
   
    real4 f = make_real4(real(0.0));

    //Bond bond; //The current bond
    
    int j; real4 posj; //The other particle      
    real3 r12;

    
    for(int b = first+threadIdx.x; b<last; b+=blockDim.x){
      /*Read bond info*/
      auto bond = bondList[b];
      j = bond.j;
      /*Bring pos of other particle*/
      posj = pos[j];
    
      /*Compute force*/
      r12 =  make_real3(posi-posj);
      //box.apply_pbc(r12);
      
      real fmod = bondForce.force(p, j, r12, bondList[b].bond_info);                  

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
      force[p] += ft;
    }

  }


  /*TODO: Launch threads only for the number of FP bonded particles*/
  template<class BondType>
  __global__ void computeBondedForcesFixedPoint(real4* __restrict__ force,
						const real4* __restrict__ pos_d,
						const uint* __restrict__ bondStart,
						const uint* __restrict__ bondEnd,
						const typename BondedForces<BondType>::BondFP* __restrict__ bondListFP,
						BondType bondForce, int N){
    
    /*Each thread in a block computes the force on particle p due to one (or several) bonds*/
    uint p = blockIdx.x*blockDim.x + threadIdx.x; //This block handles particle p
    if(p>=N) return;
    real4 pos = pos_d[p]; //Get position

    /*First and last bond indices of p in bondList*/
    uint first = bondStart[p]; 
    uint last = bondEnd[p];
   
    real4 f = make_real4(real(0.0));
    /*If I have FP bond*/
    if(first!=0xffffffff){          
      
      real3 r12;
      /*For all particles connected to me*/
      for(int b = first; b<last; b++){
	/*Retrieve bond*/
	auto bond = bondListFP[b];

	/*Compute force*/
	r12 =  make_real3(pos)-bond.pos;
	//box.apply_pbc(r12);
	
	real fmod = bondForce.force(p, 0, r12, bond.bond_info); //F = -k·(r-r0)·rvec/r
	f += make_real4(fmod*r12);
      }
      force[p] += f;
    }

  }


 
}

/*Perform an integration step*/
template<class BondType>
void BondedForces<BondType>::sumForce(){

  //BoxUtils box(L);
  if(nbonds>0){

    uint Nparticles_with_bonds = bondParticleIndex.size();
    Bonded_ns::computeBondedForces<BondType><<<Nparticles_with_bonds, TPP,
      TPP*sizeof(real4)>>>(force.d_m, pos.d_m,
			   bondStart.d_m, bondEnd.d_m,
			   bondParticleIndex.d_m, bondList.d_m,
			   bondForce);
  }
  
  if(nbondsFP>0){
    Bonded_ns::computeBondedForcesFixedPoint<BondType><<<Nblocks, Nthreads>>>(force.d_m,
								      pos.d_m,
								      bondStartFP, bondEndFP,
								      bondListFP,
								      bondForce, N);
  }
}
template<class BondType>
real BondedForces<BondType>::sumEnergy(){
  return 0;
}
template<class BondType>
real BondedForces<BondType>::sumVirial(){
  return 0;
}


/*
  Raul P. Pelaez 2016. Bonded pair forces Interactor implementation. i.e two body springs

  Interactor is intended to be a module that computes and sums the forces acting on each particle
  due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements an algorithm to compute the force between particles joined by springs.
  Sends the bondList to the GPU ordered by the first particle, and two additional arrays
    storing where the information for each particle begins and ends. Identical to the sorting
    trick in PairForces

TODO:
100- Finish the vector input of the bonds
100- Write sumEnergy
100- Implement Three body springs
*/

#include"BondedForces.h"
#include<cmath>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<set>

using namespace std;
using namespace bonded_forces_ns;

BondedForces::BondedForces(const std::vector<Bond> &bondList):
  Interactor(){

  //TODO

  init();

}
BondedForces::BondedForces(const char * readFile):
  Interactor(){

  cerr<<"Initializing Bonded Forces..."<<endl;
  params.L = L;

  nbonds = nbondsFP = 0;
  /*If some bond type number is zero, the loop will simply not be entered, and no storage will be used*/
  /*Read the bond list from the file*/
  ifstream in(readFile);
  in>>nbonds;
  if(nbonds>0){
    bondList = Vector<Bond>(nbonds*2);//Allocate 2*nbonds, see init for explication
    fori(0, nbonds){
      in>>bondList[i].i>>bondList[i].j>>bondList[i].r0>>bondList[i].k;
    }
  }
  /*Fixed point bonds*/
  in>>nbondsFP;
  if(nbondsFP>0){
    bondListFP = Vector<BondFP>(nbondsFP);
    fori(0, nbondsFP){
      in>>bondListFP[i].i;
      in>>bondListFP[i].pos.x>>bondListFP[i].pos.y>>bondListFP[i].pos.z;
      in>>bondListFP[i].r0>>bondListFP[i].k;
    }
  }

  cerr<<"\tDetected: "<<nbonds<<" particle-particle bonds and "<<nbondsFP<<" Fixed Point bonds"<<endl;
  /*Upload and init GPU*/
  init();
  cerr<<"Bonded Forces\t\tDONE!!\n\n";
}


BondedForces::~BondedForces(){}

//Criterion to sort bonds

bool bondComp(const Bond &a, const Bond &b){ return a.i<b.i;}
bool bondCompFP(const BondFP &a, const BondFP &b){ return a.i<b.i;}

//Initialize variables and upload them to GPU, init CUDA
void BondedForces::init(){
  /****************************************Pair bonds*********************************************/
  /*This algorithm is identical to the one used in PairForces to sort by cell*/
  /*First store all bonded pairs. That means i j and j i*/
  /*The first ones are readed given, the complementary have to be generated*/

  if(nbonds>0){
    fori(nbonds, 2*nbonds){
      bondList[i].i = bondList[i-nbonds].j;
      bondList[i].j = bondList[i-nbonds].i;
    
      bondList[i].k = bondList[i-nbonds].k;
      bondList[i].r0 = bondList[i-nbonds].r0;
    }
    /*There are twice as bonds now*/
    nbonds *= 2;

    /*Now sort the bondList by the first particle, i*/
    std::sort(bondList.data, bondList.data+nbonds, bondComp);

    /*We have a list of bonds ordered by its first particle, so; All the particles
      bonded with particle i=0, all particles "" i=1...*/

    /*We need additional arrays to know where in the list the bonds of particle i start
      and end*/
    /*Initially all bondStarts are 2^32-1, this value means no particles bonded*/
    bondStart = Vector<uint>(N); bondStart.fill_with(0xffffffff);
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


    
    set<uint> particlesWithBonds;
    fori(0, bondList.size()){
      particlesWithBonds.insert(bondList[i].i);
    }

    bondParticleIndex.assign(particlesWithBonds.begin(), particlesWithBonds.end());
    cerr<<"\t"<<bondParticleIndex.size()<<" particles have at least one bond"<<endl;

    /*Upload all to GPU*/
    bondParticleIndex.upload();
    bondList.upload();
    bondStart.upload();
    bondEnd.upload();
  }
  /************************************FixedPoint************************************************/
  if(nbondsFP>0){
    std::sort(bondListFP.data, bondListFP.data+nbondsFP, bondCompFP);
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
  /*Init GPU side variables*/
  initBondedForcesGPU(params);

}
/*Perform an integration step*/
void BondedForces::sumForce(){
  /*In principle these checks are not even measurable in CPU, so its fine to do it this way*/
  
  if(nbonds>0)
    //computeBondedForce(force, pos, bondStart, bondEnd, bondList, N, nbonds);
    computeBondedForce(force, pos, bondStart, bondEnd, bondParticleIndex,
		       bondList, N, bondParticleIndex.size(), nbonds);

  
  if(nbondsFP>0)
    computeBondedForceFixedPoint(force, pos,
				 bondStartFP, bondEndFP, bondListFP, N, nbondsFP);
}

float BondedForces::sumEnergy(){
  return 0.0f;
}
float BondedForces::sumVirial(){
  return 0.0f;
}

/********************************THREE BONDED FORCES**********************************/




ThreeBondedForces::ThreeBondedForces(const std::vector<ThreeBond> &bondList):
  Interactor(){


}

ThreeBondedForces::~ThreeBondedForces(){}

ThreeBondedForces::ThreeBondedForces(const char * readFile):
  Interactor(){

  cerr<<"Initializing Three Bonded Forces..."<<endl;
  params.L = L;

  nbonds = 0;
  /*Read the bond list from the file*/
  ifstream in(readFile);
  vector<vector<uint>> isInBonds(N);
  in>>nbonds;
  vector<ThreeBond> blst(nbonds); //Temporal storage for the bonds in the file

  
  cerr<<"\tDetected: "<<nbonds<<" particle-particle-particle bonds"<<endl;
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
      in>>blst[b].kspring>>blst[b].r0>>blst[b].ang;

      if(blst[b].r0>=0.5f*L){
	cerr<<"The equilibrium distance of bond "<<b<<" is too large!!"<<endl;
	cerr<<"\t This will cause unexpected behavior when computing distance using PBC!"<<endl;
      }
    }
  }


  bondList  = Vector<ThreeBond>(nbonds*3);//Allocate 3*nbonds
  bondStart = Vector<uint>(N); bondStart.fill_with(0xffFFffFF);
  bondEnd   = Vector<uint>(N); bondEnd.fill_with(0);

  
  fori(0,N){
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

  vector<uint> pwb; //Particles with bonds
  fori(0,N){
    if(bondStart[i]!=0xffFFffFF){
      pwb.push_back(i);
    }
  }

  bondParticleIndex.assign(pwb.begin(), pwb.end());
  
  nbonds *= 3; //We store all the bonds in which every particle is involved, per particle.


  bondList.upload();
  bondStart.upload();
  bondEnd.upload();
  bondParticleIndex.upload();
  
  initBondedForcesGPU(params);
  
  cerr<<pwb.size()<<" particles are involved in at least one bond."<<endl;
  
  cerr<<"Three Bonded Forces\t\tDONE!!\n\n";


}

void ThreeBondedForces::sumForce(){
  if(nbonds>0)
    computeThreeBondedForce(force, pos, bondStart, bondEnd, bondParticleIndex,
			    bondList, N, bondParticleIndex.size(), nbonds);
  cudaDeviceSynchronize();
}


float ThreeBondedForces::sumEnergy(){
  return 0.0f;
}

float ThreeBondedForces::sumVirial(){
  return 0.0f;
}

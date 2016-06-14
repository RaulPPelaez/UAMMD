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
*/

#include"BondedForces.h"
#include<cmath>
#include<iostream>
#include<fstream>
#include<algorithm>

using namespace std;

BondedForces::BondedForces(uint N, float L,
			   shared_ptr<Vector<float4>> d_pos,
			   shared_ptr<Vector<float4>> force,
			   const std::vector<Bond> &bondList):
  Interactor(N, L, d_pos, force){

  //TODO

  init();

}
BondedForces::BondedForces(uint N, float L,
			   shared_ptr<Vector<float4>> d_pos,
			   shared_ptr<Vector<float4>> force,
			   const char * readFile):
  Interactor(N, L, d_pos, force){

  params.L = L;

  /*Read the bond list from the file*/
  ifstream in(readFile);
  in>>nbonds;
  bondList = Vector<Bond>(nbonds*2);//Allocate 2*nbonds, see init for explication
  fori(0, nbonds){
    in>>bondList[i].i>>bondList[i].j>>bondList[i].r0>>bondList[i].k;
  }

  /*Upload and init GPU*/
  init();
}


BondedForces::~BondedForces(){}

//Criterion to sort bonds
bool bondComp(const Bond &a, const Bond &b){ return a.i<b.i;}

//Initialize variables and upload them to GPU, init CUDA
void BondedForces::init(){

  /*This algorithm is identical to the one used in PairForces to sort by cell*/
  /*First store all bonded pairs. That means i j and j i*/
  /*The first ones are readed given, the complementary have to be generated*/
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
  fori(0,nbonds){
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

  /*Upload all to GPU*/
  bondList.upload();
  bondStart.upload();
  bondEnd.upload();
  
  /*Init GPU side variables*/
  initBondedForcesGPU(bondStart, bondEnd, bondList, N, nbonds, params);

}
/*Perform an integration step*/
void BondedForces::sumForce(){
  computeBondedForce(force->d_m, d_pos->d_m, bondStart, bondEnd, bondList, N, nbonds);
}

float BondedForces::sumEnergy(){
  return 0.0f;
}
float BondedForces::sumVirial(){
  return 0.0f;
}

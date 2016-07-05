/*Raul P. Pelaez 2016. Short range pair forces Interactor implementation.

  Interactor is intended to be a module that computes and sums the forces acting on each particle
    due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements a Neighbour cell list search and computes the force between all neighbouring
  particle pairs. 
  The Neighbour search is implemented using hash short with cell index as hash.

TODO:
100- Generalize for any dimensions

*/
#include"PairForces.h"
#include<thread>
#include<cmath>
#include<iostream>
#include<fstream>
#include<functional>
#include<iomanip>
using namespace std;

PairForces::PairForces(pairForceType fs):
  PairForces(fs, nullForce, nullForce){}


PairForces::PairForces(pairForceType fs,
		       std::function<float(float)> cFFun,
		       std::function<float(float)> cEFun):
  customForceFunction(cFFun), customEnergyFunction(cEFun),
  rcut(gcnf.rcut), forceSelector(fs),
  Interactor()
{

  cerr<<"Initializing Pair Forces..."<<endl;
    /**Put parameters in Param struct**/
  params.rcut = rcut;

  int xcells = int(L/rcut+0.5);
  int ycells = xcells, zcells= xcells;

  params.L = L;

  params.cellSize = L/(float)xcells;

  params.xcells = xcells;
  params.ycells = ycells;
  params.zcells = zcells;

  ncells = xcells*ycells*zcells;
  params.ncells = ncells;
  cerr<<"\tNumber of cells: "<<xcells<<" "<<ycells<<" "<<zcells<<"; Total cells: "<<ncells<<endl;

  init();

  cerr<<"Pair Forces\t\tDONE!!\n\n";

}

PairForces::~PairForces(){}
//Initialize variables and upload them to GPU, init CUDA
void PairForces::init(){
  
  /*Pre compute force and energy, using force  function*/
  switch(forceSelector){
  case LJ:
    pot = Potential(forceLJ, energyLJ, 4096*params.rcut/2.5f, params.rcut);
    break;
  case CUSTOM:
    pot = Potential(customForceFunction, customEnergyFunction, 4096*params.rcut/2.5f, params.rcut);
    break;
  case NONE:
    pot = Potential(nullForce, nullForce, 2, params.rcut);
    break;
  default:
    cerr<<"NON RECOGNIZED POTENTIAL SELECTED!!"<<endl;
    exit(1);
  }

  sortPos   = Vector<float4>(N); sortPos.fill_with(make_float4(0.0f)); sortPos.upload(); 

  /*Temporal storage for the enrgy and virial per particle*/
  energyArray = Vector<float>(N); energyArray.fill_with(0.0f); energyArray.upload();
  virialArray = Vector<float>(N); virialArray.fill_with(0.0f); virialArray.upload();

  
  cellIndex = Vector<uint>(N+1); cellIndex.upload();
  particleIndex= Vector<uint>(N+1); particleIndex.upload();
  cellStart        = Vector<uint>(ncells); cellStart.upload();
  cellEnd          = Vector<uint>(ncells); cellEnd.upload();

  initPairForcesGPU(params,
		    pot.getForceData(), pot.getEnergyData(), pot.getSize(),
		    cellStart, cellEnd, particleIndex, ncells,
		    sortPos, N);

}

/*** CONSTRUCT NEIGHBOUR LIST ***/
void PairForces::makeNeighbourList(){
  /*Compute cell id of each particle*/
  calcCellIndex(d_pos->d_m, cellIndex, particleIndex, N);

  /*Sort the particle indices by hash (cell index)*/
  sortCellIndex(cellIndex, particleIndex, N);
  /*Reorder positions by cell index and construct cellStart and cellEnd*/
  reorderAndFind(sortPos,
		 cellIndex, particleIndex,
		 cellStart, cellEnd, params.ncells,
		 d_pos->d_m, N); 
}


void PairForces::sumForce(){
  /*** CONSTRUCT NEIGHBOUR LIST ***/
  makeNeighbourList();
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  computePairForce(sortPos,
	       force->d_m, 
	       cellStart, cellEnd, 
	       particleIndex,
	       N);
}

float PairForces::sumEnergy(){
  /*** CONSTRUCT NEIGHBOUR LIST ***/
  //makeNeighbourList();
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
 return computePairEnergy(sortPos,
			  energyArray, 
			  cellStart, cellEnd, 
			  particleIndex,
			  N);

}
float PairForces::sumVirial(){
  /*** CONSTRUCT NEIGHBOUR LIST ***/
  //makeNeighbourList();
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  float v =  computePairVirial(sortPos,
			       virialArray, 
			       cellStart, cellEnd, 
			       particleIndex,
			       N);

  return v/(3.0f*L*L*L);
}




//Force between two particles, depending on square distance between them
// this function is only called on construction, so it doesnt need to be optimized at all
float forceLJ(float r2){
  float invr2 = 1.0f/(r2);
  float invr6 = invr2*invr2*invr2;		 
  float invr8 = invr6*invr2;		 
  
  float fmod = -48.0f*invr8*invr6+24.0f*invr8;

  return fmod;
}
float energyLJ(float r2){
  float invr2 = 1.0f/r2;
  float invr6 = invr2*invr2*invr2;
  float E =  2.0f*invr6*(invr6-1.0f);

  return E;
}
float nullForce(float r2){return 0.0f;}


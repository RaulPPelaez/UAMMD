/*Raul P. Pelaez 2016. Short range pair forces Interactor implementation.

  Interactor is intended to be a module that computes and sums the forces acting on each particle
    due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements a Neighbour cell list search and computes the force between all neighbouring
  particle pairs. 
  The Neighbour search is implemented using hash short with a 30 bit morton code (from the cell coordinates) as hash.

  The Neighbour list is constructed as follows:
  
  1-Compute a hash for each particle based on its cell. Store in particleHash, also fill particleIndex with the index of each particle (particleIndex[i] = i)
  2-Sort particleIndex based on particleHash (sort by key). This way the particles in a same cell are one after the other in particleIndex. The Morton hash also improves the memory acces patter in the GPU.
  3-Fill cellStart and cellEnd with the indices of particleIndex in which a cell starts and ends. This allows to identify where all the [indices of] particles in a cell are in particleIndex, again, one after the other.
  
  The transversal of this cell list is done by transversing, for each particle, the 27 neighbour cells of that particle's cell.

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
using namespace pair_forces_ns;

PairForces::PairForces(pairForceType fs,
		       std::function<float(float)> cFFun,
		       std::function<float(float)> cEFun):
  Interactor(),
  sortPos(N), particleHash(N), particleIndex(N),
  energyArray(N), virialArray(N),
  customForceFunction(cFFun), customEnergyFunction(cEFun),
  rcut(gcnf.rcut), forceSelector(fs)
{
  pairForcesInstances++;

  cerr<<"Initializing Pair Forces..."<<endl;
    /**Put parameters in Param struct**/
  if(pairForcesInstances>1){ cerr<<"ERROR: Only one PairForces instance is allowed!!!"<<endl; exit(1);}
  params.rcut = rcut;

  int xcells = int(L/rcut+0.5);
  int ycells = xcells, zcells= xcells;

  params.L = L;

  params.cellSize = L/(float)xcells;

  params.cellDim.x = xcells;
  params.cellDim.y = ycells;
  params.cellDim.z = zcells;

  ncells = xcells*ycells*zcells;
  params.ncells = ncells;
  cerr<<"\tNumber of cells: "<<xcells<<" "<<ycells<<" "<<zcells<<"; Total cells: "<<ncells<<endl;

  init();

  cerr<<"Pair Forces\t\tDONE!!\n\n";

}

PairForces::~PairForces(){
  /*It is a good idea to free all the memory used on destruction, at least on the GPU side.
    CUDA appears to have some issues if you dont do it*/
  sortPos.freeMem();
  energyArray.freeMem();
  virialArray.freeMem();
  particleHash.freeMem();
  particleIndex.freeMem();
  cellStart.freeMem();
  cellEnd.freeMem(); 
}
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

  sortPos.fill_with(make_float4(0.0f)); sortPos.upload();

  /*Temporal storage for the enrgy and virial per particle*/
  energyArray.fill_with(0.0f); energyArray.upload();
  virialArray.fill_with(0.0f); virialArray.upload();

  
  particleHash.fill_with(0);  particleHash.upload();
  particleIndex.fill_with(0); particleIndex.upload();
  cellStart    = Vector<uint>(ncells); cellStart.fill_with(0);     cellStart.upload();
  cellEnd      = Vector<uint>(ncells); cellEnd.fill_with(0);       cellEnd.upload();
  
  initPairForcesGPU(params,
		    pot.getForceTexture(), pot.getEnergyTexture(),
		    cellStart, cellEnd, particleIndex, ncells,
		    sortPos, pos, N);
  
  cudaDeviceSynchronize();
}

/*** CONSTRUCT NEIGHBOUR LIST ***/
void PairForces::makeNeighbourList(){
  makeCellList(pos, sortPos, particleIndex, particleHash, cellStart, cellEnd, N, ncells);
}


void PairForces::sumForce(){
  static int steps = 0;
  steps++;

  /*** CONSTRUCT NEIGHBOUR LIST ***/
  makeNeighbourList();

  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  computePairForce(sortPos,
		   force, 
		   cellStart, cellEnd, 
		   particleIndex,
		   N);

   // force.download();
   // float4 sumforce = std::accumulate(force.begin(), force.end(), make_float4(0));

   // cout<<sumforce.x<<" "<<sumforce.y<<" "<<sumforce.z<<endl; 
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

uint PairForces::pairForcesInstances = 0;

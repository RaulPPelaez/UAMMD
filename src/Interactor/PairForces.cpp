/*Raul P. Pelaez 2016. Short range pair forces Interactor implementation.

  Interactor is intended to be a module that computes and sums the forces acting on each particle
    due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements a Neighbour cell list search and computes the force between all neighbouring
  particle pairs. 
  The Neighbour search is implemented using hash short with cell index as hash.

*/
#include"PairForces.h"
#include<thread>
#include<cmath>
#include<iostream>
#include<fstream>
#include<functional>
#include<iomanip>
using namespace std;

PairForces::PairForces(uint N, float L, float rcut,
		       shared_ptr<Vector<float4>> d_pos,
		       shared_ptr<Vector<float4>> force,
		       pairForceType fs, std::function<float(float)> customForceFunction):
  rcut(rcut),forceSelector(fs), customForceFunction(customForceFunction),  
  Interactor(N, L, d_pos, force){

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
  cerr<<"Number of cells: "<<xcells<<" "<<ycells<<" "<<zcells<<"; Total cells: "<<ncells<<endl;

  init();

}
PairForces::~PairForces(){}
//Initialize variables and upload them to GPU, init CUDA
void PairForces::init(){
  cerr<<"Initializing...";
  
  /*Pre compute force, using force  function*/
  switch(forceSelector){
  case LJ:
    pot = Potential(forceLJ, 4096, params.rcut);
    break;
  case CUSTOM:
    pot = Potential(customForceFunction, 4096, params.rcut);
    break;
  case NONE:
    pot = Potential(nullForce, 2, params.rcut);
    break;
  default:
    cerr<<"NON RECOGNIZED POTENTIAL SELECTED!!"<<endl;
    exit(1);
  }

  sortPos   = Vector<float4>(N); sortPos.fill_with(make_float4(0.0f)); sortPos.upload(); 

  cellIndex = Vector<uint>(N+1); cellIndex.upload();
  particleIndex= Vector<uint>(N+1); particleIndex.upload();
  cellStart        = Vector<uint>(ncells); cellStart.upload();
  cellEnd          = Vector<uint>(ncells); cellEnd.upload();

  initPairForcesGPU(params, pot.getData(), pot.getSize(), cellStart, cellEnd, particleIndex, ncells, sortPos, N);

  cerr<<"\tDONE!!"<<endl;
}
/*Perform an integration step*/
void PairForces::sumForce(){
  /*** CONSTRUCT NEIGHBOUR LIST ***/
  /*Compute cell id of each particle*/
  calcCellIndex(d_pos->d_m, cellIndex, particleIndex, N);

  /*Sort the particle indices by hash (cell index)*/
  sortCellIndex(cellIndex, particleIndex, N);
  /*Reorder positions by cell index and construct cellStart and cellEnd*/
  reorderAndFind(sortPos,
		 cellIndex, particleIndex,
		 cellStart, cellEnd, params.ncells,
		 d_pos->d_m, N); 
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  computePairForce(sortPos,
	       force->d_m, 
	       cellStart, cellEnd, 
	       particleIndex,
	       N);
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
float nullForce(float r2){return 0.0f;}


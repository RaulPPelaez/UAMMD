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
#include"PairForcesAlt.h"
#include<thread>
#include<cmath>
#include<iostream>
#include<fstream>
#include<functional>
#include<iomanip>
#include<random>
using namespace std;
using namespace pair_forces_alt_ns;

PairForcesAlt::PairForcesAlt(pairForceType fs):
  Interactor(),
  rcut(gcnf.rcut), forceSelector(fs)
{
  pairForcesInstances++;

  cerr<<"Initializing Pair Forces..."<<endl;
    /**Put parameters in Param struct**/
  if(pairForcesInstances>1){ cerr<<"ERROR: Only one PairForcesAlt instance is allowed!!!"<<endl; exit(1);}
  maxNPerCell = N>64?64:N;

  params.rc = rcut;
  params.rmax = 1.2*rcut;
  
  int xcells = int(L/params.rmax+0.5);
  int ycells = xcells, zcells= xcells;

  params.L = L;

  params.invCellSize = 1.0f/(L/(float)xcells);

  params.cellDim = make_int3(xcells, ycells, zcells);

  ncells = xcells*ycells*zcells;
  //params.ncells = ncells;
  cerr<<"\tNumber of cells: "<<xcells<<" "<<ycells<<" "<<zcells<<"; Total cells: "<<ncells<<endl;

  params.errorFlag = nullptr;
  
  init();

  cerr<<"Pair Forces\t\tDONE!!\n\n";

}

PairForcesAlt::~PairForcesAlt(){


}
//Initialize variables and upload them to GPU, init CUDA
void PairForcesAlt::init(){
  
  /*Pre compute force and energy, using force  function*/
  switch(forceSelector){
  case LJ:
    pot = Potential(forceLJ, energyLJ, 4096*params.rc/2.5f, params.rc);
    break;
  case NONE:
    pot = Potential(nullForce, nullForce, 2, params.rc);
    break;
  default:
    cerr<<"NON RECOGNIZED POTENTIAL SELECTED!!"<<endl;
    exit(1);
  }

  sortPos = Vector4(N); sortPos.fill_with(make_float4(0.0f)); sortPos.upload();
  old_pos = Vector4(N); old_pos.fill_with(make_float4(0xffFFffFF)); old_pos.upload(); 
  
  cellIndex = Vector<uint>(N); cellIndex.fill_with(0); cellIndex.upload();
  particleIndex = Vector<uint>(N); particleIndex.fill_with(0); particleIndex.upload();
  cellSize    = Vector<uint>(ncells); cellSize.fill_with(0);     cellSize.upload();
  //cellEnd      = Vector<uint>(ncells); cellEnd.fill_with(0);       cellEnd.upload();
  
  Nneigh = Vector<uint>(N); Nneigh.fill_with(0);       Nneigh.upload();
  NBL = Vector<uint>(N*maxNPerCell); NBL.fill_with(0); NBL.upload();
  CELL = Vector<uint>(ncells*maxNPerCell); CELL.fill_with(0); CELL.upload();
  
  initPairForcesAltGPU(params, pos, sortPos, NBL,  N, maxNPerCell);

}

/*** CONSTRUCT NEIGHBOUR LIST ***/
void PairForcesAlt::makeNeighbourList(){
  static uint counter = 0;
  counter++;

  if(checkBinningCells(old_pos, pos, N, 0.5f*(params.rc-params.rmax))){
    while(!makeNeighbourListGPU2(cellIndex, cellSize, particleIndex,
				 params.rmax,
				 CELL, ncells,
				 NBL, Nneigh,
				 pos, old_pos, sortPos,
				 N, maxNPerCell)){
      /*Then the number of neighbours must be increased*/
      cerr<<"\nChanging maxNPerCell to: ";
      maxNPerCell += 32;
      NBL.freeMem();
      CELL.freeMem();
      NBL = Vector<uint>(N*maxNPerCell); NBL.GPUfill_with(0.0f);
      CELL = Vector<uint>(ncells*maxNPerCell); CELL.GPUfill_with(0);
      rebindGPU(maxNPerCell, N , NBL);
      cerr<<maxNPerCell<<" "<<counter<<endl;
   // Nneigh.download();
   //  std::sort(Nneigh.begin(), Nneigh.end());
   //  cerr<<"nneigh: "<<Nneigh[N-1]<<endl;

      rebindGPU(maxNPerCell, N, NBL);
    }

    // cerr<<"\rUpdate cells steps:  "<<counter<<"        ";
    // counter = 0;
  }
  cudaDeviceSynchronize();


  
  // cellSize.download();
  // std::sort(cellSize.begin(), cellSize.end());
  // cerr<<cellSize[ncells-1]<<endl;
  //NBL.download();

  //  fori(0,NBL.size()) if(NBL[i]>N) cerr<<i<<" "<<NBL[i]<<endl;
  // exit(0);
}


void PairForcesAlt::sumForce(){
  /*** CONSTRUCT NEIGHBOUR LIST ***/
  this->makeNeighbourList();


  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  computePairForce(NBL, Nneigh, particleIndex,
		   pos, sortPos, force,
		   N, maxNPerCell);

}

float PairForcesAlt::sumEnergy(){
  return 0.0f;
}
float PairForcesAlt::sumVirial(){
  return 0.0f;
}

uint PairForcesAlt::pairForcesInstances = 0;

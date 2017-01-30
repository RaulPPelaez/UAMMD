/*Raul P. Pelaez 2016. Short range pair forces Interactor implementation.

  Interactor is intended to be a module that computes and sums the forces acting on each particle
    due to some interaction, like and external potential or a pair potential.

  The positions and forces must be provided, they are not created by the module.

  This module implements a Neighbour cell list search and computes the force between all neighbouring
  particle pairs. 
  The Neighbour search is implemented using hash short with a 30 bit morton code (from the cell coordinates) as hash.

  The Neighbour list is constructed as follows:

  You need: 
      -particleIndex (size N)
      -particleHash (size N)
      -cellStart (size ncells)
      -cellEnd   (size ncells)


  1-Compute a hash for each particle based on its cell (a Morton hash following a Z-order curve). Store in particleHash, also fill particleIndex with the index of each particle (particleIndex[i] = i)
  2-Sort particleIndex based on particleHash (sort by key). This way the particles in a same cell are one after the other in particleIndex. The Morton hash also improves the memory acces pattern in the GPU.
  3-Fill cellStart and cellEnd with the indices of particleIndex in which a cell starts and ends. This allows to identify where all the [indices of] particles in a cell are in particleIndex, again, one after the other.
  
  The transversal of this cell list is done by transversing, for each particle, the 27 neighbour cells of that particle's cell.

  I have found that this list works best when it is reconstructed in all steps (as oposed to having a cell size > rcut and checking each step if the particles have moved enough to reconstruct the list).
  The construction of a neighbour list in addition to the cell list also appears to be slower than just transversing the 27 neighbour cells.

  In PairForcesAlt there is a version following the more standard approach (see e.g hoomd) constructing a cell list /neighbour list with a carefully coalesced data structure ( cellList has ncells*maxNperCell size i.e). That version constructs a cell list AND a neighbour list, but the cell size > rcut and it is only needed to be redone after a bunch of steps.
  The list construction is much more expensive in that case, but it is masked given that it is not computed each step. On the other hand, currently transversing the list here is actually faster than in ParForcesAlt.
  Overall, this list appears to be much faster than the one in PairForcesAlt for N>10K. And that is not taking into account that in the other case the Z-ordering must be done outside the neighbour list construction logic, whereareas in this version the ordering is part of the algorithm. So that is an additional benefit.

  On the other hand, the lack of an explicit neighbour list makes this algorithm a little more troublesome to communicate with the outside, because the transversing logic must be shipped along with the data structures (there is not an array simply containing N lists of neighbours). And besides, more particles are checked innecessarily.

  I expect to see even further improvement (specially on low N) when the version with several threads per particle of the transversing is done. 

  Also I do not know how well suited for multi gpu this algorithm will be when dealing with ghost particles. Probably there wont be much difference.

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
#include<set>
using namespace std;
using namespace pair_forces_ns;
/*TODO: Study the needsUpdateGPU/verlet list, it appears that it is not beneficial at all*/
PairForces::PairForces(pairForceType fs,
		       std::function<real(real)> cFFun,
		       std::function<real(real)> cEFun):
  Interactor(),
  sortPos(N),/* oldPos(N),*/ particleHash(N), particleIndex(N),
  energyArray(N), virialArray(N),
  customForceFunction(cFFun), customEnergyFunction(cEFun),
  rcut(gcnf.rcut), forceSelector(fs){
  my_instance = pairForcesInstances;
  pairForcesInstances++;
  
  cerr<<"Initializing Pair Forces..."<<endl;
    /**Put parameters in Param struct**/
  params.rcut_pot = gcnf.rcut;
  params.rcut = rcut;

  int3 cellDim = make_int3(L/rcut);
  if(L.z==real(0.0)) cellDim.z = 1;
  
  params.L = L;
  params.N = N;
  
  params.cellSize = L/make_real3(cellDim);

  params.cellDim = cellDim;


  ncells = cellDim.x*cellDim.y*cellDim.z;

  if(cellDim.x== 2 || cellDim.y == 2 || cellDim.z == 2)
    cerr<<"WARNING: Only 2 cells in one direction can cause unexpected behavior, as a particle can interact with another both in the same box and in a periodic box at the same time.   # . | o # . -> .-o and  o-."<<endl;
  if(ncells == 1)
    cerr<<"WARNING: Using Pair forces with just one cell is equivalent to the Nbody Forces module, and performance will be much better with the latter"<<endl;
  
  params.ncells = ncells;
  cerr<<"\tNumber of cells: "<<cellDim.x<<" "<<cellDim.y<<" "<<cellDim.z<<"; Total cells: "<<ncells<<endl;

  cerr<<"\tCut-off distance: "<<params.rcut_pot<<endl;
  init();

  cerr<<"Pair Forces\t\tDONE!!\n\n";

}

PairForces::~PairForces(){
  sortPos.freeMem();
  //  oldPos.freeMem();
  energyArray.freeMem();
  virialArray.freeMem();
  particleHash.freeMem();
  particleIndex.freeMem();
  cellStart.freeMem();
  cellEnd.freeMem(); 
}
//Initialize variables and upload them to GPU, init CUDA
void PairForces::init(){

  uint ntypes = 0;

  vector<uint> types(N);
  fori(0,N) types[i] = (uint)(pos[i].w+0.5);
  /*Discard duplicates*/
  set<uint> typeset(types.begin(), types.end());
  ntypes = typeset.size();
  cerr<<"\t"<<ntypes<<" Paticle types detected"<<endl;
  if(ntypes>1 && gcnf.sigma != real(1.0)){

    cerr<<"\tWARNING: gcnf.sigma should be 1.0 when using multiple types if you are setting different sigmas as potential parameters!"<<endl;

  }
  /*Pre compute force and energy, using force  function*/
  switch(forceSelector){
  case LJ:
    cerr<<"\tUsing LJ potential"<<endl;
    pot = Potential(forceLJ, energyLJ, 4096*params.rcut/real(2.5), params.rcut, ntypes);
    break;
  case CUSTOM:
    cerr<<"\tUsing custom potential"<<endl;
    pot = Potential(customForceFunction, customEnergyFunction, 4096*params.rcut/real(2.5), params.rcut, ntypes);
    break;
  case NONE:
    pot = Potential(nullForce, nullForce, 2, params.rcut, ntypes);
    break;
  default:
    cerr<<"NON RECOGNIZED POTENTIAL SELECTED!!"<<endl;
    exit(1);
  }

  sortPos.fill_with(make_real4(0.0)); sortPos.upload();
  //oldPos.fill_with(make_real4(0xffFFffFF)); oldPos.upload();

  /*Temporal storage for the enrgy and virial per particle*/
  energyArray.fill_with(0.0); energyArray.upload();
  virialArray.fill_with(0.0); virialArray.upload();

  
  particleHash.fill_with(0);  particleHash.upload();
  particleIndex.fill_with(0); particleIndex.upload();
  cellStart    = Vector<uint>(ncells); cellStart.fill_with(0);     cellStart.upload();
  cellEnd      = Vector<uint>(ncells); cellEnd.fill_with(0);       cellEnd.upload();


  /*Set texture references in params for constant memory*/
  params.texForce = pot.getForceTexture();
  params.texEnergy = pot.getEnergyTexture();

  params.texPos = pos.getTexture();
  params.texSortPos = sortPos.getTexture();
  params.texCellStart = cellStart.getTexture();
  params.texCellEnd = cellEnd.getTexture();


  params.ntypes = ntypes;
  params.potParams = pot.getPotParams();
  
  /*Upload parameters to GPU, put params in paramsGPU(global memory) and constant memory*/
  initGPU(params, paramsGPU, N, pot.getSize()*sizeof(float));

  /*The constant memory version of Params is mine now*/
  pairForcesConstantMemoryOwner = my_instance;
  
  cudaDeviceSynchronize();
}
void PairForces::setPotParam(uint i, uint j, real2 param){
  pot.setPotParam(i,j,param);
}
/*** CONSTRUCT NEIGHBOUR LIST ***/
void PairForces::makeNeighbourList(){
  makeCellList(pos, sortPos, //oldPos,
	       particleIndex, particleHash, cellStart, cellEnd, N, ncells);
}


void PairForces::sumForce(){
  static int steps = 0;
  steps++;
  /*Ensure my parameters are in constant memory at the .cu*/
  if(pairForcesConstantMemoryOwner != my_instance){
    pairForcesConstantMemoryOwner = my_instance;
    updateParamsFromGPU(paramsGPU);
  }
  
    
    
  /*** CONSTRUCT NEIGHBOUR LIST ***/
  //  if(needsUpdateGPU(pos, oldPos, (params.rcut-params.rcut_pot)*0.5, N))
    makeNeighbourList();
  // else{
  //   reorderPosGPU(sortPos, pos, particleIndex, N);

  // }
 

  
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  computePairForce(sortPos,
		   force, 
		   cellStart, cellEnd, 
		   particleIndex,
		   N);
  // if(steps%100==0){
    // force.download();
    // double4 sumforced = make_double4(0,0,0,0);


    // fori(0,N){
    //   sumforced.x += (double)force[i].x;
    //   sumforced.y += (double)force[i].y;
    //   sumforced.z += (double)force[i].z;
    // }
    // //real4 sumforce = std::accumulate(force.begin(), force.end(), make_real4(0));
    // cerr<<sumforced.x<<" "<<sumforced.y<<" "<<sumforced.z<<endl;
  
}

real PairForces::sumEnergy(){
  if(pairForcesConstantMemoryOwner != my_instance){
    pairForcesConstantMemoryOwner = my_instance;
    updateParamsFromGPU(paramsGPU);
  }

  /*** CONSTRUCT NEIGHBOUR LIST ***/
  makeNeighbourList();

  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
 return computePairEnergy(sortPos,
			  energyArray, 
			  cellStart, cellEnd, 
			  particleIndex,
			  N);

}
real PairForces::sumVirial(){
  if(pairForcesConstantMemoryOwner != my_instance){
    pairForcesConstantMemoryOwner = my_instance;
    updateParamsFromGPU(paramsGPU);
  }

  /*** CONSTRUCT NEIGHBOUR LIST ***/
  makeNeighbourList();
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  real v =  computePairVirial(sortPos,
			       virialArray, 
			       cellStart, cellEnd, 
			       particleIndex,
			       N);

  return v/(real(3.0)*L.x*L.y*L.z);
}



template<class Transversable>
void PairForces::sumCustom(Transversable t){
  t.print();
  if(pairForcesConstantMemoryOwner != my_instance){
    pairForcesConstantMemoryOwner = my_instance;
    updateParamsFromGPU(paramsGPU);
  }

  /*** CONSTRUCT NEIGHBOUR LIST ***/
  makeNeighbourList();

  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  return computeWithListGPU(t, 
			    cellStart, cellEnd, 
			    particleIndex,
			    N);

}


//Force between two particles, depending on square distance between them
// this function is only called on construction, so it doesnt need to be optimized at all
//Distance is in units of sigma
real forceLJ(real r2){
  real invr2 = 1.0/(r2);
  real invr = 1.0/sqrt(r2);
  real invr6 = invr2*invr2*invr2;
  real invr8 = invr6*invr2;

  real invrc13 = pow(1.0/gcnf.rcut, 13);
  real invrc7 = pow(1.0/gcnf.rcut, 7);
  
  real fmod = -48.0*invr8*invr6 + 24.0*invr8;  
  real fmodcorr = 48.0*invr*invrc13 - 24.0*invr*invrc7;

  if(gcnf.rcut>2.0*gcnf.sigma)
  //(f(r)-f(rcut))/r
    return fmod+fmodcorr;
  else
    return fmod;
}
real energyLJ(real r2){
  real r = sqrt(r2);
  real invr2 = 1.0/r2;
  real invr6 = invr2*invr2*invr2;
  //real E =  2.0f*invr6*(invr6-1.0f);
  //potential as u(r)-(r-rcut)*f(rcut)-r(rcut) 
  real E = 2.0*(invr6*(invr6-1.0));
  real Ecorr = 2.0*(
		      -(r-gcnf.rcut)*(-24.0*pow(1.0/gcnf.rcut, 13)+12.0*pow(1.0/gcnf.rcut, 7))
		      -pow(1.0/gcnf.rcut, 6)*(pow(1.0/gcnf.rcut, 6)-1.0));
  if(gcnf.rcut>2.0*gcnf.sigma)
    return E+Ecorr;
  else
    return E;
}
real nullForce(real r2){return 0.0;}

uint PairForces::pairForcesInstances = 0;
uint PairForces::pairForcesConstantMemoryOwner = 0;

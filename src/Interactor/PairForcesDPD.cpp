#include"PairForcesDPD.h"



PairForcesDPD::PairForcesDPD(): PairForces(LJ), rngCPU(gcnf.seed){

  cerr<<"Initializing DPD Submodule..."<<endl;
  gamma = 10.0f;
  paramsDPD.gamma = gamma;
  paramsDPD.noiseAmp = sqrt(2.0f*gamma*gcnf.T/gcnf.dt);


  sortVel = Vector4(N); sortVel.fill_with(make_float4(0.0f)); sortVel.upload();
  
  initPairForcesDPDGPU(paramsDPD, sortVel, N);

  /*Warmup rng*/
  fori(0, 1000) seed = rngCPU.next();


  cerr<<"DPD Submodule\tDONE!!"<<endl;
}


/*** CONSTRUCT NEIGHBOUR LIST ***/
void PairForcesDPD::makeNeighbourListDPD(){
  /*Compute cell id of each particle*/
  calcCellIndex(pos, cellIndex, particleIndex, N);

  /*Sort the particle indices by hash (cell index)*/
  sortCellIndex(cellIndex, particleIndex, N);
  /*Reorder positions and velocities by cell index and construct cellStart and cellEnd*/
  reorderAndFindDPD(sortPos, sortVel,
   		    cellIndex, particleIndex,
   		    cellStart, cellEnd, params.ncells,
   		    pos, vel, N); 
}

void PairForcesDPD::sumForce(){

  /*** CONSTRUCT NEIGHBOUR LIST ***/
  makeNeighbourListDPD();

  /*Move the seed to the next step*/
  seed = rngCPU.next();
  
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  computePairForceDPD(sortPos, force, vel,
		      cellStart, cellEnd, 
		      particleIndex,
		      N, seed);
}


float PairForcesDPD::sumEnergy(){ return 0.0f;}
float PairForcesDPD::sumVirial(){ return 0.0f;}

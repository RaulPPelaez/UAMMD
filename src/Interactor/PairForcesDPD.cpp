#include"PairForcesDPD.h"



PairForcesDPD::PairForcesDPD(): PairForces(LJ){

  cerr<<"Initializing DPD Submodule..."<<endl;
  gamma = 10.0f;
  paramsDPD.gamma = gamma;
  paramsDPD.noiseAmp = sqrt(6.0f*gamma*gcnf.T/gcnf.dt);
  
  initPairForcesDPDGPU(paramsDPD);
  cerr<<"DPD Submodule\tDONE!!"<<endl;
}


void PairForcesDPD::sumForce(){
  /*** CONSTRUCT NEIGHBOUR LIST ***/
  PairForces::makeNeighbourList();
  /*** COMPUTE FORCES USING THE NEIGHBOUR LIST***/
  computePairForceDPD(sortPos, force, vel,
		      cellStart, cellEnd, 
		      particleIndex,
		      N);
}


float PairForcesDPD::sumEnergy(){ return 0.0f;}
float PairForcesDPD::sumVirial(){ return 0.0f;}

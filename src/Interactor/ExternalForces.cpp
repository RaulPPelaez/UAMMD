/*
  Raul P. Pelaez 2016. External Force interactor.
  Applies some custom force function, see ExternalForcesGPU.cu, to each particle
*/

#include"ExternalForces.h"
#include<cmath>
#include<iostream>

using namespace std;
using namespace external_forces_ns;

ExternalForces::ExternalForces():
  Interactor(){
  cerr<<"Initializing External Forces..."<<endl;
  /*Upload and init GPU*/
  init();
  cerr<<"External Forces\t\tDONE!!\n\n";
}


ExternalForces::~ExternalForces(){}

//Initialize variables and upload them to GPU, init CUDA
void ExternalForces::init(){
  
  params.L = L;
  initGPU(params);
}
/*Perform an integration step*/
void ExternalForces::sumForce(){
  computeExternalForce(force, pos, N);
}

float ExternalForces::sumEnergy(){
  return 0.0f;
}
float ExternalForces::sumVirial(){
  return 0.0f;
}

/*
  Raul P. Pelaez 2016. NBody Force Interactor

  Computes the interaction between all pairs in the system. Currently only gravitational force
*/

#include"NBodyForces.h"
#include<cmath>
#include<iostream>
#include<fstream>
#include<algorithm>

using namespace std;
using namespace nbody_ns;

NBodyForces::NBodyForces():
  Interactor(){
  //TODO

  init();

}

NBodyForces::~NBodyForces(){}



void NBodyForces::init(){
  
}

void NBodyForces::sumForce(){

  computeNBodyForce(force, pos, N);
}

real NBodyForces::sumEnergy(){
  return 0.0;
}
real NBodyForces::sumVirial(){
  return 0.0;
}

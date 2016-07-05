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

NBodyForces::NBodyForces():
  Interactor(){
  //TODO

  init();

}

NBodyForces::~NBodyForces(){}



void NBodyForces::init(){
  
}

void NBodyForces::sumForce(){

  computeNBodyForce(force->d_m, d_pos->d_m, N);
}

float NBodyForces::sumEnergy(){
  return 0.0f;
}
float NBodyForces::sumVirial(){
  return 0.0f;
}

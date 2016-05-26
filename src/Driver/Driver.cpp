/*
Raul P. Pelaez 2016. MD simulator using Interactor and Integrator, example of usage.


NOTES:
The idea is to use either Integrator or Interactor in another project as a module.

Once initialized this classes will perform a single task very fast as black boxes:

Integrator uploads the positions according to the velocities, forces and current positions.
Interactor computes the pair forces using the current positions according to the selected potential

The idea is for Integrator to control the positions and velocities and for Interactor to control the forces. Communicating each variable when needed. So if you need the vel. in the force computing you can pass it when computing the force and modify the force function accordingly.

*/

#include "Driver.h"

//Constructor
Driver::Driver(uint N, float L, float rcut, float dt): N(N){
  /*Create the position array*/
  pos = Vector<float4>(N,true);
  /*Start in a cubic lattice*/
  cubicLattice(pos.data, L, N);
  /*Once done upload to GPU*/
  pos.upload();

  /*Initialize the modules*/
  /*Interactor needs the positions, it handles the forces, LJ is an enum for the force type*/
  interactor = new Interactor(N, L, rcut, &pos, LJ);
  /*Integrator needs the positions and forces addresses, it handles the velocities*/
  integrator = new Integrator(&pos, interactor->getForce(), N, L, dt);


}
  
//Perform one step
void Driver::update(){
  /*First step of the integrator*/
  integrator->updateFirstStep();
  interactor->compute_force();
  /*This fucntion could be overloaded to do nothing if the integrator is only one step!*/
  integrator->updateSecondStep();
}

//Integrator handles the writing
void Driver::write(bool block){
  integrator->write(block);
}
//Read an initial configuratio nfrom fileName, TODO
void Driver::read(const char *fileName){
  ifstream in(fileName);
  float r,c,l;
  in>>l;
  fori(0,N){
    in>>pos[i].x>>pos[i].y>>pos[i].z>>r>>c;
  }
  in.close();
  pos.upload();
}

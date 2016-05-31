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
  pos = Vector<float4>(N, true);
  /*Start in a cubic lattice*/
  cubicLattice(pos.data, L, N);
  /*Once done upload to GPU*/
  pos.upload();

  /*The force is handled outside for convinience*/
  force = Vector<float4>(N); force.fill_with(make_float4(0.0f)); force.upload();

  /****Initialize the modules*******/
  /*This is the simulation construction, where you choose integrator and force evaluators*/

  /*Interactor needs the positions, it handles the forces, LJ is an enum for the force type,
   if set to CUSTOM, the next parameter is the name of a force function float(float r2)*/
  interactor = make_shared<PairForces>(N, L, rcut, make_shared<Vector<float4>>(pos),
				       make_shared<Vector<float4>>(force), LJ);
  /*Integrator needs the positions and forces addresses, it handles the velocities*/
  /*Integrator is an abstract virtual base clase that has to be overloaded for each new integrator
    . This mantains retrocompatibility, and allows for new integrators to be added without changes*/
  /*To use one or another, just instanciate them as in here. Using a two step velocity verlet integrator i.e.*/
  integrator = make_shared<TwoStepVelVerlet>(make_shared<Vector<float4>>(pos),
					     make_shared<Vector<float4>>(force), N, L, dt);
  /*You can add several different interactors to an integrator*/
  integrator->addInteractor(interactor);

}
  
//Perform one step
void Driver::update(){
  /*First step of the integrator*/
  integrator->update();
  // interactor->compute_force();
  // /*This fucntion could be overloaded to do nothing if the integrator is only one step!*/
  // integrator->updateSecondStep();
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

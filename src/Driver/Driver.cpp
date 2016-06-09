/*
Raul P. Pelaez 2016. MD simulator using Interactor and Integrator, example of usage.

NOTES:
The idea is to mix implementations of Integrator and Interactor to construct a simulation. 
For example create a TwoStepVelVerlet integrator and add a PairForces interactor with LJ to create a lennard jonnes gas MD simulation.


Once initialized this classes will perform a single task very fast as black boxes:

Integrator uploads the positions according to the velocities, forces and current positions.
Interactor computes the pair forces using the current positions according to the selected potential

The idea is for Integrator to control the positions and velocities and for Interactor to control the forces. Communicating each variable when needed. So if you need the vel. in the force computing you can pass it to your implementation of Interactor and use it in the force function accordingly.

*/

#include "Driver.h"
#include "Interactor/BondedForces.h"
#define RANDESP (rand()/(float)RAND_MAX)

//#define BROWNIAN_EXAMPLE

//Constructor
Driver::Driver(uint N, float L, float rcut, float dt): N(N){
  /*Create the position array*/
  pos = Vector<float4>(N, true);
  pos.fill_with(make_float4(0.0f));
  /*Start in a cubic lattice*/
  cubicLattice2D(pos.data, L, N);

  /*Id some particles (the ones joined by springrs in the p.bonds example file*/
  /*The output of the program will print x y z 0.56 c, where c is pos.w+1*/
  pos[0].w = 1;
  pos[1].w = 1;
  pos[2].w = 1;
  pos[25].w = 2;
  pos[132].w = 2;
  pos[7].w = 1;
   
  /*Once done upload to GPU*/
  pos.upload();

  /*The force is handled outside for convinience*/
  force = Vector<float4>(N); force.fill_with(make_float4(0.0f)); force.upload();

  /*Here there are two examples of simulation constructions, toggle them with the BROWNIAN DEFINE
    First one with particles interacting via LJ pot and qith some of them joined by springs.
    The integrator is a two step vel verlet. */
  #ifndef BROWNIAN_EXAMPLE
  /****Initialize the modules*******/
  /*This is the simulation construction, where you choose integrator and force evaluators*/

  /*Interactor needs the positions, and modifies the forces and any additional parameter
    particular to the interactor. In the case of PairForces, LJ is an enum for the force type,
   if set to CUSTOM, the next parameter is the name of a force function float(float r2)*/
  shared_ptr<Interactor> interactor2 =
    make_shared<PairForces>(N, L, rcut, make_shared<Vector<float4>>(pos),
			    make_shared<Vector<float4>>(force), LJ);
  /*In the case of BondedForces, only pos, force and the name
    of a file with the bond information is needed. Alternatively, 
    you can pass a vector<Bond> containing all the bonds*/
  shared_ptr<Interactor> interactor =
    make_shared<BondedForces>(N, L,
 			      make_shared<Vector<float4>>(pos),
 			      make_shared<Vector<float4>>(force), "p.bonds");
  
  /*Integrator needs the positions and forces addresses, N, L, dt and any additional
    parameter particular to the integrator. L must be passed even if the box is infinite, just pass 0 i.e.
  /*Integrator is an abstract virtual base clase that has to be overloaded for each new integrator
    . This mantains retrocompatibility, and allows for new integrators to be added without changes*/
  /*To use one or another, just instanciate them as in here. Using a two step velocity verlet integrator i.e.*/
  integrator = make_shared<TwoStepVelVerlet>(make_shared<Vector<float4>>(pos),
					     make_shared<Vector<float4>>(force), N, L, dt);

  /*You can add several interactors to an integrator as such*/
  integrator->addInteractor(interactor2);
  integrator->addInteractor(interactor);

#endif
  /*The second example is a Brownian dynamics simulation with non interacting particles and a 
    shear flow acting on the x direction as a function of the y position.*/
#ifdef BROWNIAN_EXAMPLE
  cubicLattice2D(pos.data, L, N);
  // fori(0,N){
  //   float x=1, y=1;
  //   while(x*x+y*y > 1){
  //     x = (RANDESP*2.0f-1.0f);
  //     y = (RANDESP*2.0f-1.0f);
  //   }
  //   pos[i].x = x*L;
  //   pos[i].y = y*L;
  //   pos[i].z = 0.0f;
  //  }

  pos.upload();
  
  D = Vector<float4>(4);
  D.fill_with(make_float4(0.0f));
  D[0].x = 1;
  D[1].y = 1;
  D[2].z = 1;
  D.upload();
  
  
  K = Vector<float4>(4);
  K.fill_with(make_float4(0.0f));
  K[0] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
  K.upload();
  
  integrator = make_shared<BrownianEulerMaruyama>(make_shared<Vector<float4>>(pos),	      
   						  make_shared<Vector<float4>>(force),
   						  make_shared<Vector<float4>>(D),
   						  make_shared<Vector<float4>>(K),
   						  N, L, dt);
#endif  
  
  
}
  
//Perform one step
void Driver::update(){
  integrator->update();
}

//Integrator handles the writing
void Driver::write(bool block){
  integrator->write(block);
}
//Read an initial configuration from fileName, TODO
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

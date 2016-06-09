/*Raul P. Pelaez 2016. Integrator abstract class base implementation

  Integrator is intended to be a separated module that handles the update of positions and applies forces on each particle via Interactors

  It takes care of keeping the positions updated.

  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk

  This is an abstract class that should be derived to implement new integrators, Integrator itself cannot be instanciated.

 Currently Implemented integrators:
   1. Velocity Verlet
   2. Brownian Dynamics Euler Maruyama (WIP)
   2. Brownian Dynamics Euler Maruyama with Hydrodynamics (WIP)
*/

#include"Integrator.h"

//Basic constructor
Integrator::Integrator(){
  writeThread=NULL;

  steps = 0;
}
Integrator::~Integrator(){}

//Constructor to be called in the initialization list of the derived class
Integrator::Integrator(shared_ptr<Vector<float4>> pos,
		       shared_ptr<Vector<float4>> force, uint N, float L, float dt):  
  pos(pos), force(force),
  N(N), dt(dt), L(L){
  writeThread=NULL;
  steps = 0;
}


//Write a step to disk using a separate thread
void Integrator::write(bool block){
  /*Wait for the last write operation to finish*/
  if(writeThread) writeThread->join();
  /*Bring pos from GPU*/
  pos->download();
  /*Wait for copy to finish*/
  cudaDeviceSynchronize();
  /*Query the write operation to another thread*/
  writeThread = new std::thread(write_concurrent, pos->data, L, N);
  if(block) writeThread->join();
}

//This function writes a step to disk
void write_concurrent(float4 *pos, float L, uint N){
  cout<<"#L="<<L*0.5<<";\n";
  fori(0,N){
    cout<<pos[i].x<<" "<<pos[i].y<<" "<<pos[i].z<<" 0.56 "<<(int)(pos[i].w+1)<<"\n";
  }
}


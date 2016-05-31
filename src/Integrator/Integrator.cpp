/*Raul P. Pelaez 2016. Integrator class

  Integrator is intended to be a separated module that handles the update of positions given the forces

  It takes care of creating the velocities and keep the positions updated.
  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk
  
  TODO:
   Maybe the velocities should be outside the module, handled as the positions.

 */
#include"Integrator.h"


Integrator::Integrator(){
  writeThread=NULL;

  steps = 0;
}
Integrator::~Integrator(){}
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
    cout<<pos[i].x<<" "<<pos[i].y<<" "<<pos[i].z<<" 0.56 1\n";
  }
}


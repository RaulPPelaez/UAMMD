/*Raul P. Pelaez 2016. Integrator class base implementation

  Integrator is intended to be a separated module that handles the update of positions and applies forces on each particle via Interactors

  It takes care of keeping the positions updated.

  The positions must be provided, they are not created by the module.
  Also takes care of writing to disk

  This is an abstract class that should be derived to implement new integrators, Integrator itself cannot be instanciated.

 Currently Implemented integrators:
   1. Velocity Verlet NVE
   2. Velocity Verlet NVT with BBK thermostat
   3. Brownian Dynamics Euler Maruyama
   4. Brownian Dynamics Euler Maruyama with Hydrodynamics
*/

#include"Integrator.h"

Integrator::~Integrator(){
  if(writeThread.joinable())
    writeThread.join();
}

//Constructor to be called in the initialization list of the derived class
Integrator::Integrator():
  N(gcnf.N), dt(gcnf.dt), L(gcnf.L){
  steps = 0;
}


//Write a step to disk using a separate thread
void Integrator::write(bool block){
  /*Wait for the last write operation to finish*/

  if(this->writeThread.joinable()){
    this->writeThread.join();
  }
  /*Bring pos from GPU*/
  pos.download();
  /*Wait for copy to finish*/
  cudaDeviceSynchronize();
  /*Query the write operation to another thread*/
  this->writeThread =  std::thread(write_concurrent, pos.data);
  
  if(block && this->writeThread.joinable()){
    this->writeThread.join();
  }
  
}

//This function writes a step to disk
int write_concurrent(float4* posdata){
  float L = gcnf.L;
  uint N = gcnf.N;
  cout<<"#L="<<L*0.5<<";\n";
  fori(0,N){
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" 0.56 "<<(int)(posdata[i].w+1)<<"\n";
  }
  
  return 0;
  // fori(0,N)
  //   printf("%f\t%f\t%f\t", pos[i].x+ 2.5f*( (i==0)?1.0f:-1.0f), pos[i].y, pos[i].z);
  // printf("\n");
}


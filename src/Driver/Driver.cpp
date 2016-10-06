/*
Raul P. Pelaez 2016. MD simulator using Interactor and Integrator handler.

NOTES:
The idea is to mix implementations of Integrator and Interactor to construct a simulation. 
For example create a VerletNVT integrator and add a PairForces interactor with LJ to create a lennard jonnes MD simulation in the NVT ensemble.

Once initialized this classes will perform a single task very fast as black boxes:

Integrator uploads the positions according to the forces and current positions, using anything else it needs internally.
Interactor computes forces acting on each particle. For that it has default access to the positions. You can implement additional needs in each case, like velocities for example.
*/

#include "Driver.h"
#define RANDESP (rand()/(float)RAND_MAX)

//Constructor
Driver::Driver(): step(0){
  /*Initialize pos and force arrays*/

}

void Driver::setParameters(){

  uint N = gcnf.N;
  if(pos.size() != N){
    pos = Vector4(N);
    pos.fill_with(make_float4(0.0f));
    pos.upload();
  }
  
  force = Vector4(N);
  force.fill_with(make_float4(0.0f));
  force.upload();


}

//Perform the simulation steps
void Driver::run(uint nsteps, bool relax){

  Timer tim;
  tim.tic();
  /*Simulation*/
  fori(0,nsteps){
    step++;
    
    integrator->update();
    if(!relax){
      if(i%gcnf.print_steps==0 && gcnf.print_steps >= 0 )
	this->write(); //Writing is done in parallel, is practically free if the interval is big enough
      
      
      if(i%gcnf.measure_steps==0 && gcnf.measure_steps>0)
	for(auto m: measurables)
	  m->measure();
    }
  }
  cerr<<tim.toc()<<endl;
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
  fori(0,gcnf.N){
    in>>pos[i].x>>pos[i].y>>pos[i].z>>r>>c;
  }
  in.close();
  pos.upload();
}


Driver::~Driver(){
  
}




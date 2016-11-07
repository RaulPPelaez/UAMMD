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

//Constructor
Driver::Driver(): step(0){
  /*Initialize pos and force arrays*/

}

void Driver::setParameters(){

  uint N = gcnf.N;
  if(pos.size() != N){
    pos = Vector4(N);
    pos.fill_with(make_real4(real(0.0)));
    pos.upload();
  }
  if(force.size()!=N){
    force = Vector4(N);
    force.fill_with(make_real4(real(0.0)));
    force.upload();
  }

  grng = Xorshift128plus(gcnf.seed);

  cerr<<endl;
  cerr<<"Sigma: "<<gcnf.sigma<<endl;
  cerr<<"Box size: "<<gcnf.L<<endl;
  cerr<<"Number of particles: "<<gcnf.N<<endl;

  cerr<<"Time step: "<<gcnf.dt<<endl;
  if(gcnf.print_steps>0){
    cerr<<"\tPrint every: "<<gcnf.print_steps<<endl;
    cerr<<"\tTime between steps in file: "<<gcnf.dt*gcnf.print_steps<<endl;
  }
  cerr<<"Random seed: "<<std::hex<<"0x"<<gcnf.seed<<"ull"<<endl;
  cerr<<std::dec<<endl;
}

//Perform the simulation steps
void Driver::run(uint nsteps, bool relax){
  if(relax)cerr<<"Running "<<nsteps<<" relaxation steps..."<<endl;
  Timer tim;
  tim.tic();
  /*Simulation*/
  fori(0,nsteps){
    step++;
    
    integrator->update();
    if(!relax){
      if(i%gcnf.print_steps==0 && gcnf.print_steps >= 0 )
	this->writer.write(); //Writing is done in parallel, is practically free if the interval is big enough
      
      if(i%gcnf.measure_steps==0 && gcnf.measure_steps>0)
	for(auto m: measurables)
	  m->measure();
    }
  }
  cerr<<"Run time: "<<tim.toc()<<"s"<<endl;
  gcnf.nsteps += nsteps;
}

//Read an initial configuration from fileName, TODO
void Driver::read(const char *fileName){
  ifstream in(fileName);
  real r,c,l;
  in>>l;
  fori(0,gcnf.N){
    in>>pos[i].x>>pos[i].y>>pos[i].z>>r>>c;
  }
  in.close();
  pos.upload();
}


Driver::~Driver(){
  cudaDeviceSynchronize();
  /*Free the global arrays manually*/
  pos.freeMem();
  force.freeMem();
  vel.freeMem(); 
}


Writer::~Writer(){
  if(writeThread.joinable())
    writeThread.join();
}


//Write a step to disk using a separate thread
void Writer::write(bool block){
  /*Wait for the last write operation to finish*/

  if(this->writeThread.joinable()){
    this->writeThread.join();
  }
  /*Wait for any GPU work to be done*/
  cudaDeviceSynchronize();
  /*Bring pos from GPU*/
  pos.download();
  /*Wait for copy to finish*/
  cudaDeviceSynchronize();
  /*Query the write operation to another thread*/
  this->writeThread =  std::thread(&Writer::write_concurrent, this);

  /*Wait if needed*/
  if(block && this->writeThread.joinable()){
    this->writeThread.join();
  }
  
}

//TODO: generalize format somehow, and allow for any other global array to be written
//This function writes a step to disk
void Writer::write_concurrent(){
  real3 L = gcnf.L;
  uint N = gcnf.N;
  real4 *posdata = pos.data;
  cout<<"#Lx="<<L.x*0.5f<<";Ly="<<L.y*0.5f<<";Lz="<<L.z*0.5f<<";\n";
  fori(0,N){

    uint type = (uint)(posdata[i].w+0.5);
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<(type==0?0.5f:1.0f)<<" "<<type<<"\n";
  }
  //  cout<<make_real3(posdata[0])<<" "<<make_real3(posdata[1]);
}

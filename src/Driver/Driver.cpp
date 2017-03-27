/*
Raul P. Pelaez 2016. Driver implementation. Controls the flow of the simulation. See Driver.h
*/

#include "Driver.h"
#include<iomanip>
//Constructor does nothing
Driver::Driver(): step(0){
}

void Driver::setParameters(){
  /*Initialize pos and force if needed*/
  if(pos.size() != gcnf.N && pos.size() > 0) gcnf.N = pos.size();
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
  /*Initialize RNG*/
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

  if(gcnf.L.z==real(0.0)) gcnf.D2 = true;
  
  GlobalConfigGPU gcnfGPU_m;
  gcnfGPU_m.L = gcnf.L;
  gcnfGPU_m.invL = 1.0/gcnf.L;
  gcnfGPU_m.N = gcnf.N;
  gcnfGPU_m.dt = gcnf.dt;
  gcnfGPU_m.T = gcnf.T;
  gcnfGPU_m.D2 = gcnf.D2;
  
  gpuErrchk(cudaMemcpyToSymbol(gcnfGPU, &gcnfGPU_m, sizeof(GlobalConfigGPU)));


  
}

/*Forward the simulation state nsteps·dt in time*/
void Driver::run(uint nsteps, bool relax){
  if(relax)cerr<<"Running "<<nsteps<<" relaxation steps..."<<endl;
  
  Timer tim, timfps; /*Counter of total time and FPS helper timer*/
  tim.tic(); timfps.tic();
  uint fps_steps = 0;
  /*Simulation*/
  fori(0,nsteps){
    current_step = i;
    step++;
    fps_steps++;
    /**********FPS computation is done each second******/
    if(timfps.toc()>1.0f){
      float fps = fps_steps/timfps.toc();
      cerr<<"\rComputing step: "<<step<<setprecision(5)<<"   FPS: "<<fps;

      int remaining = (int((nsteps-i)/fps));
      if(remaining>0){
	int hours   = remaining/3600;
	int minutes = remaining/60-hours*60;
	int seconds = remaining-minutes*60-hours*3600;
		
	cerr<<"   ETA: ";
	if(hours){
	  cerr<<hours<<"h:";
	  if(minutes)cerr<<minutes<<"m:";
	}
	else if(minutes)
	  if(minutes)cerr<<minutes<<"m:";	

	cerr<<seconds<<"s             ";
	fps_steps = 0;
      }
      timfps.tic();      
    }
    /*FORWARD THE SIMULATION dt*/
    integrator->update();
    if(!relax){
      /*Print if needed*/
      if(i%gcnf.print_steps==0 && gcnf.print_steps > 0 )
	this->writer.write(); //Writing is done in parallel, is practically free if the interval is big enough
      /*MEasure observables if needed*/
      if(i%gcnf.measure_steps==0 && gcnf.measure_steps>0)
	for(auto m: measurables)
	  m->measure();
    }
  }
  cerr<<"     Run time: "<<tim.toc()<<"s                                           "<<endl;
  gcnf.nsteps += nsteps;
}

Driver::~Driver(){
  cudaDeviceSynchronize();
  /*Wait for any writing process to finish*/
  writer.synchronize();
  /*Free the global arrays manually*/
  pos.freeMem();
  force.freeMem();
  vel.freeMem(); 
}


Writer::~Writer(){
  if(writeThread.joinable())
    writeThread.join();
}

void Writer::synchronize(){
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
// //This function writes a step to disk
// void Writer::write_concurrent(){
//   real3 L = gcnf.L;
//   uint N = gcnf.N;
//   real4 *posdata = pos.data;
//   cout<<"#Lx="<<L.x*0.5f<<";Ly="<<L.y*0.5f<<";Lz="<<L.z*0.5f<<";\n";
//   fori(0,N){    
//     uint type = (uint)(posdata[i].w+0.5);
//     cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<0.5f*gcnf.sigma<<" "<<type<<"\n";
//   }
//   cout<<flush;
// }

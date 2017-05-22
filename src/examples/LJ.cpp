/* Raul P.Pelaez 2016- Simulation Config
   
   Implementation of the SimulationConfig class. 

   The constructor of this class sets all the parameters, creates the initial configuration and constructs the simulation.


   This is a simulation of a LJ liquid using a neighbour list and a verlet NVT integrator.

 */

#include "SimulationConfig.h"
#include<random>
#include<iomanip>

SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
    
  Timer tim; tim.tic();
  
  gcnf.T = 1.0;
  gcnf.gamma = 1.0;
  
  gcnf.L = make_real3(128);

  gcnf.N = pow(2,20);
  gcnf.dt = 0.01;
  

  gcnf.rcut = 2.5;
  
  gcnf.nsteps1 = 100000;
  gcnf.nsteps2 = 0;
  gcnf.print_steps = -1;
  gcnf.measure_steps = -1;


  gcnf.seed = 0xffaffbfDEADBULL^time(NULL);
  pos = initLattice(gcnf.L, gcnf.N, fcc); //Start in a fcc lattice

  setParameters();

  pos.upload();
  
  integrator = make_shared<VerletNVT>();

  auto lj = make_shared<PairForces<CellList, Potential::LJ>>();
  lj->setPotParams(0,0,
		   {1.0f /*epsilon*/, 1.0f /*sigma*/, 2.5f /*rcut*/, true/*shift?*/});
  integrator->addInteractor(lj);

  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;
  
  tim.tic();

  run(gcnf.nsteps1);  
  
  
  /*********************************End of simulation*****************************/
  double total_time = tim.toc();
  
  cerr<<"\nMean step time: "<<setprecision(5)<<(double)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nSimulation time: "<<setprecision(5)<<total_time<<"s"<<endl;  

}

void Writer::write_concurrent(){
  real3 L = gcnf.L;
  uint N = gcnf.N;
  real4 *posdata = pos.data;
  cout<<"#Lx="<<L.x*0.5f<<";Ly="<<L.y*0.5f<<";Lz="<<L.z*0.5f<<";\n";
  fori(0,N){    
    uint type = (uint)(posdata[i].w+0.5);
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<0.5f*gcnf.sigma<<" "<<type<<"\n";
  }
  cout<<flush;
}

/* Raul P.Pelaez 2016- Simulation Config
   
   Implementation of the SimulationConfig class. 

   The constructor of this class sets all the parameters, creates the initial configuration and constructs the simulation.
 */

#include "SimulationConfig.h"
#include<random>
#include<iomanip>

SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
    
  Timer tim; tim.tic();
  
  gcnf.E = 0.0;
  gcnf.T = 0.3;
  gcnf.gamma = 1.0;
  
  gcnf.L = make_real3(32);

  gcnf.N = pow(2,14);
  gcnf.dt = 0.01;
  

  gcnf.rcut = 2.5;//pow(2, 1/6.)*gcnf.sigma;//2.5*gcnf.sigma;//  //rcut = 2.5*sigma -> biggest sigma in the system;
  
  gcnf.nsteps1 = 100000;
  gcnf.nsteps2 = 0;
  gcnf.print_steps = 500;
  gcnf.measure_steps = -1;


  gcnf.seed = 0xffaffbfDEADBULL;//^time(NULL);
  pos = initLattice(gcnf.L, gcnf.N, fcc); //Start in a simple cubic lattice

  int N = gcnf.N;

  setParameters();


  // pos.fill_with(make_real4(1));
  // pos.upload();
  // Vector<int> tmp(N);
  // NBody_ns::SimpleCountingTransverser tr(tmp.d_m); 

  // auto nbody = make_shared<NBodyForces<NBody_ns::SimpleCountingTransverser>>(tr);
  // tim.tic();
  // int counter = 0;
  // while(tim.toc()<1){
  //   nbody->sumForce();
  //   cudaDeviceSynchronize();
  //   counter++;
  // }
  // cerr<<counter<<endl;
  // cerr<<(counter/tim.toc())<<endl<<endl;
  // tmp.download(10);
  // fori(0,10) cerr<<tmp[i]<<endl;  
  // exit(1);

  
  pos.upload();
  
  integrator = make_shared<VerletNVT>();

  auto interactor = make_shared<PairForces<CellList>>();  
  integrator->addInteractor(interactor);

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

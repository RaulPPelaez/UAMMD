/*Raul P. Pelaez 2017

  UAMMD input file example.
  
  A brownian hydrodynamics simulation:
     A system of LJ colloids in a falling drop with hydrodynamic interactions via Positively Split Edwald method.

  The system starts in a cube, moves out of it a few time steps using NVT MD and then starts interacting hydrodynamically, falling due to a gravity added by GravityFunctor, forming a drop.

*/
#include "SimulationConfig.h"

/*This functor is passed to ExternalForces and adds to each particle a force in the z direction, each step*/
struct GravityFunctor{
  inline __device__ real4 operator()(const real4 &pos, int i){
    return make_real4(0,0,0.1f,0);
    
  }
};

SimulationConfig::SimulationConfig(int argc, char* argv[]): Driver(){
  Timer tim; tim.tic();

  /*Set parameters*/
  gcnf.T = 1.0;
  
  gcnf.L = make_real3(50, 50, 128);

  gcnf.N = pow(2,14);
  gcnf.dt = 0.01;


  gcnf.nsteps1 = 1000;
  gcnf.nsteps2 = 100000;
  gcnf.print_steps = 200;  

  /*Seed for the random number generation*/
  gcnf.seed = 0xffaffbfDEADBULL^time(NULL);

  /*Start with an FCC lattice*/
  pos = initLattice(make_real3(32), gcnf.N, fcc);

  /*Fix parameters*/
  setParameters();

  /*Upload positions to gpu*/
  pos.upload();

  /*A pair forces module using LJ potential*/
  auto lj = make_shared<PairForces<CellList, Potential::LJ>>(2.5/*rcut*/);
  /*Only one type of interaction*/
  lj->setPotParams(0,0,
		  {1/*epsilon*/, 1 /*Sigma*/, 2.5 /*rcut*/, true /*shift?*/});


  /*Relax the system with an NVT MD integrator*/
  integrator = make_shared<VerletNVT>();
  /*Add the interactor*/
  integrator->addInteractor(lj);

  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;

  tim.tic();
  
  /*Run some relaxation steps*/
  run(gcnf.nsteps1, true);
    

  /*Change the integrator and run the simulation*/
  real rh = 1.0; /*Hydrodynamic radius*/
  real vis = 1.0;/*viscosity*/
  Matrixf K(3,3);/*Shear matrix*/
  K.fill_with(0);

  auto gravity = make_shared<ExternalForces<GravityFunctor>>(GravityFunctor());
  integrator = make_shared<BrownianHydrodynamicsEulerMaruyama>(K, vis, rh, PSE);
  integrator->addInteractor(lj);
  integrator->addInteractor(gravity);
  
  /*Run the simulation*/
  run(gcnf.nsteps2);
  
  /*********************************End of simulation*****************************/
  double total_time = tim.toc();

  cerr<<"\nMean step time: "<<setprecision(5)<<(double)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nSimulation time: "<<setprecision(5)<<total_time<<"s"<<endl;

}
/*Change the output format here, in this function, the updated positions will be available 
 in CPU. Writing is done in parallel, while the simulation is still running.
*/
void Writer::write_concurrent(){
  real3 L = gcnf.L;
  uint N = gcnf.N;
  real4 *posdata = pos.data;
  cout<<"#Lx="<<L.x*0.5f<<";Ly="<<L.y*0.5f<<";Lz="<<L.z*0.5f<<";\n";
  fori(0,N){
    uint type = (uint)(posdata[i].w+0.5);
    cout<<posdata[i].x<<" "<<posdata[i].y<<" "<<posdata[i].z<<" "<<.5f<<" "<<type<<"\n";
  }
  cout<<flush;
}


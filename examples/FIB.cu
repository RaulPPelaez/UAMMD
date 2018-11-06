/*Raul P. Pelaez 2018. Fluctuating Immerse Boundary for Brownian Dynamics with Hydrodynamic Interactions demo.

See BDHI/FIB.cuh for more info.
 */


#include"uammd.cuh"
#include"Integrator/BDHI/FIB.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>

using namespace std;
using namespace uammd;



int main(int argc, char *argv[]){

  if(argc==1){
    std::cerr<<"Run with: ./fib 14 32 0.01 50000 200 1 1"<<std::endl;
    exit(1);
  }

  int N = pow(2,atoi(argv[1]));//atoi(argv[1]));
  
  auto sys = make_shared<System>();

  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(N, sys);

  Box box(std::stod(argv[2]));
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    
    auto initial =  initLattice(box.boxSize, N, sc);
    fori(0,N){
      pos.raw()[i] = initial[i];
      pos.raw()[i].w = 0;//sys->rng().uniform(0,1)>std::stod(argv[6])?0:1;
    }    
  }
  

  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  
  ofstream out("kk");
  
  double hydrodynamicRadius =  std::stod(argv[7]);
  
  BDHI::FIB::Parameters par;
  par.temperature = std::stod(argv[6]);
  par.viscosity = 1.0;
  par.hydrodynamicRadius =  hydrodynamicRadius;
  par.dt = std::stod(argv[3]);
  par.box = box;
  par.scheme=BDHI::FIB::MIDPOINT; //Use simple midpoint scheme
  auto bdhi = make_shared<BDHI::FIB>(pd, sys, par);
   
  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();

  bdhi->forwardTime();
  bdhi->forwardTime();
  Timer tim;
  tim.tic();
  int nsteps = std::atoi(argv[4]);
  int printSteps = std::atoi(argv[5]);
  //Run the simulation
  forj(0,nsteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    bdhi->forwardTime();
    //Write results
    if(j%printSteps==0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      //continue;
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      //This allows to access the particles with the starting order so the particles are written in the same order
      // even after a sorting
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      
      out<<"#"<<endl;
      real3 p;
      fori(0,N){
	real4 pc = pos.raw()[sortedIndex[i]];
	p = make_real3(pc);
	int type = pc.w;
	out<<p<<" "<<hydrodynamicRadius<<" "<<type<<endl;
      }
    }
    //Sort the particles every few steps
    //It is not an expensive thing to do really.
    if(j%500 == 0){
      pd->sortParticles();
    }
  }
  
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
  //sys->finish() will ensure a smooth termination of any UAMMD module.
  sys->finish();

  return 0;
}
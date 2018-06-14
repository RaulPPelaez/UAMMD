/*Raul P. Pelaez 2017. A short range forces example.

This file contains a good example of how UAMMD works and how to configure and launch a simulation.

Runs a Brownian Hydrodynamics simulation with particles starting in a periodic box at low temperature.
  
Needs cli input arguments with a system size, etc, look for "argv"

Or just run: ./a.out 14 32 0.001 1000 10000 0.0 0.1
for a quick test

You can visualize the reuslts with superpunto

*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
//#include"Integrator/VerletNVT.cuh"
#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
// #include"Integrator/BDHI/BDHI_Cholesky.cuh"
//#include"Integrator/BDHI/BDHI_Lanczos.cuh"
#include"Integrator/BDHI/BDHI_PSE.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/Potential/Potential.cuh"

#include<thrust/sort.h>

#include"utils/InitialConditions.cuh"
#include<fstream>


using namespace uammd;
using namespace std;


int main(int argc, char *argv[]){

  if(argc==1){
    std::cerr<<"Run with: ./a.out 12 64 0.01 10000 100 0.0 0.1 1"<<std::endl;
    exit(1);
  }

  int N = pow(2,atoi(argv[1]));//atoi(argv[1]));
  cerr<<N<<endl;
  //UAMMD System entity holds information about the GPU and tools to interact with the computer itself (such as a loging system). All modules need a System to work on.
  
  auto sys = make_shared<System>();

  //Modules will ask System when they need a random number (i.e for seeding the GPU RNG).
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  //ParticleData stores all the needed properties the simulation will need.
  //Needs to start with a certain number of particles, which can be changed mid-simulation
  //If UAMMD is to be used as a plugin for other enviroment or custom code, ParticleData should accept references to
  // properties allocated and handled by the user, this is a non-implemented work in progress as of now though.
  auto pd = make_shared<ParticleData>(N, sys);

  //Some modules need a simulation box (i.e PairForces for the PBC)
  Box box(std::stod(argv[2]));
  //Initial positions
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    
  //Start in a fcc lattice, pos.w contains the particle type
    //auto initial =  initLattice(box.boxSize, N, sc);
    
    fori(0,N){
      pos.raw()[i] = make_real4(sys->rng().uniform3(-box.boxSize.x*0.5, box.boxSize.x*0.5), 0);
      //Type of particle is stored in .w
      pos.raw()[i].w = sys->rng().uniform(0,1)>std::stod(argv[6])?0:1;
    }    
  }
  

  //Modules can work on a certain subset of particles if needed, the particles can be grouped following any criteria
  //The builtin ones will generally work faster than a custom one. See ParticleGroup.cuh for a list
  
  //A group created with no criteria will contain all the particles  
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  
  ofstream out("kk");
  
  BDHI::PSE::Parameters par;
  par.temperature = std::stod(argv[7]);
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = std::stod(argv[3]);
  par.box = box;
  par.tolerance = std::stod(argv[10]);
  par.psi=std::stod(argv[8]);
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::PSE>>(pd, pg, sys, par);
   
  /*
  VerletNVT::Parameters par;
  par.temperature = std::stod(argv[7]);
  par.dt = std::stod(argv[3]);  
  auto bdhi = make_shared<VerletNVT>(pd, pg, sys, par);

  */
  using PairForces = PairForces<Potential::LJ>;

  //This is the general interface for setting up a potential
  auto pot = make_shared<Potential::LJ>(sys);
  {
    //Each Potential describes the pair interactions with certain parameters.
    //The needed ones are in InputPairParameters inside each potential, in this case:
    Potential::LJ::InputPairParameters par;
    par.epsilon = 1.0;
    par.shift = false;    
        
    par.sigma = 2.0;
    par.cutOff =par.sigma*pow(2, 1/6.);
    
    pot->setPotParameters(0, 0, par);
  }

  PairForces::Parameters params;
  params.box = box;  //Box to work on
  auto pairforces = make_shared<PairForces>(pd, pg, sys, params, pot);

  if(atoi(argv[9])>0)bdhi->addInteractor(pairforces);
  
  //You can issue a logging event like this, a wide variety of log levels exists (see System.cuh).
  //A maximum log level is set in System.cuh, every logging event with a level superior to the max will result in
  // absolutely no overhead, so dont be afraid to write System::DEBUGX log calls.
  sys->log<System::MESSAGE>("RUNNING!!!");

  //Ask ParticleData to sort the particles in memory!
  //It is a good idea to sort the particles once in a while during the simulation
  //This can increase performance considerably as it improves coalescence.
  //Sorting the particles will cause the particle arrays to change in order and (possibly) address.
  //This changes will be informed with signals and any module that needs to be aware of such changes
  //will acknowedge it through a callback (see ParticleData.cuh).
  pd->sortParticles();
        
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
	out<<p<<" "<<0.5*(type==1?2:1)<<" "<<type<<endl;
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
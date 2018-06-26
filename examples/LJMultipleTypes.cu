/*Raul P. Pelaez 2017. A short range forces example.

This file contains a good example of how UAMMD works and how to configure and launch a simulation.

It describes a LJ liquid simulation in a periodic box.

This example showcases the performance when using a "large" number of particle types.
It does this by setting particles types randomly from 0 to an input number and then configuring the potential
to handle each particle pair. In this case all particle pairs have the same parameters, but the potential is oblivious to this.

Seek argv[ to see what each parameter does
Run: ./ljMult 14 32 0.01 1.0 10000 300 100

for a quick test

You can visualize the reuslts with superpunto
  
 */

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/VerletNVT.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>


using namespace uammd;
using namespace std;


int main(int argc, char *argv[]){

  if(argc<7){
    std::cerr<<"ERROR, I need some parameters!!\nTry to run me with:\n./a.out 14 32 0.01 1.0 10000 300 300"<<std::endl;
    exit(1);
  }
  int N = pow(2, atoi(argv[1]));//atoi(argv[1]));

  //UAMMD System entity holds information about the GPU and tools to interact with the computer itself (such as a loging system). All modules need a System to work on.
  
  auto sys = make_shared<System>();

  //Modules will ask System when they need a random number (i.e for seeding the GPU RNG).
  ullint seed = 0xf31337Bada55D00dULL;
  sys->rng().setSeed(seed);

  //ParticleData stores all the needed properties the simulation will need.
  //Needs to start with a certain number of particles, which can be changed mid-simulation
  //If UAMMD is to be used as a plugin for other enviroment or custom code, ParticleData should accept references to
  // properties allocated and handled by the user, this is a non-implemented work in progress as of now though.
  auto pd = make_shared<ParticleData>(N, sys);

  //Some modules need a simulation box (i.e PairForces for the PBC)
  Box box(std::stod(argv[2]));
  //Initial positions
  int ntypes = std::atoi(argv[7]);
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);    

    auto initial =  initLattice(box.boxSize*std::stod(argv[4]), N, fcc);
    
    //Start in a cubic lattice, pos.w contains the particle type
    //auto initial = cubicLattice(box.boxSize, N);
    fori(0,N){
      pos.raw()[i] = initial[i];
      //Type of particle is stored in .w
     
      pos.raw()[i].w = sys->rng().next32()%ntypes;
    }    
  }
  
  //Modules can work on a certain subset of particles if needed, the particles can be grouped following any criteria
  //The builtin ones will generally work faster than a custom one. See ParticleGroup.cuh for a list
  
  //A group created with no criteria will contain all the particles  
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  
  
  ofstream out("kk");
  
  //Some modules need additional parameters, in this case VerletNVT needs dt, temperature...
  //When additional parameters are needed, they need to be supplied in a form similar to this:
  using NVT = VerletNVT::GronbechJensen;
  
  NVT::Parameters par;
  par.temperature = 0.1;
  par.dt = std::stod(argv[3]);
  par.viscosity = 1.0/(6*M_PI);  
  auto verlet = make_shared<NVT>(pd, pg, sys, par);

  //Modules working with pairs of particles usually ask for a Potential object
  //PairForces decides if it should use a neighbour list or treat the system as NBody,
  //You can force the use of a certain neighbour list passing its name as a second template argument
  {
    using PairForces = PairForces<Potential::LJ>;

    //This is the general interface for setting up a potential
    auto pot = make_shared<Potential::LJ>(sys);
    {
      //Each Potential describes the pair interactions with certain parameters.
      //The needed ones are in InputPairParameters inside each potential, in this case:
      Potential::LJ::InputPairParameters par;
      par.epsilon = 1.0;
      par.shift = false;    
      par.sigma = 1;
      par.cutOff = 2.5*par.sigma;
      //All interaction parameters are the same, though the potential does not know this.
      fori(0,ntypes){
	forj(0,ntypes){
	  pot->setPotParameters(i, j, par);
	}
      }
    }

    PairForces::Parameters params;
    params.box = box;  //Box to work on
    auto pairforces = make_shared<PairForces>(pd, pg, sys, params, pot);  
    //You can add as many modules as necessary
    verlet->addInteractor(pairforces);
  }  


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
  int nsteps = std::atoi(argv[5]);
  int printSteps = std::atoi(argv[6]);
  //Run the simulation
  forj(0,nsteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    verlet->forwardTime();

    //Write results
    if(j%printSteps==1)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      //continue;
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
      real3 p;
      fori(0,N){
	real4 pc = pos.raw()[sortedIndex[i]];
	p = box.apply_pbc(make_real3(pc));
	int type = pc.w;
	out<<p<<" "<<0.5*pow(2,1/6.)<<" "<<type<<endl;
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

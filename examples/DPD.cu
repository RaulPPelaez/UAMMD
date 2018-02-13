/*Raul P. Pelaez 2017. A short Dissipative Particle Dynamics example.

This file contains a good example of how UAMMD works and how to configure and launch a simulation.


Particles start in a random uniform configuration. They evolve following DPD integration.
The parameters are set to reproduce the results in [1]. You can check the gdr and see that it perfectly reproduces Figure 3 in [1].



  
Needs cli input arguments with a system size, etc, look for "argv"

Or just run: ./a.out 100000 5000 0.01
for a quick test

You can visualize the reuslts with superpunto

References:
[1] On the numerical treatment of dissipative particle dynamics and related systems. Leimkuhler and Shang 2015. https://doi.org/10.1016/j.jcp.2014.09.008  
 */

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/VerletNVE.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/Potential/DPD.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>


using namespace uammd;
using namespace std;

int main(int argc, char *argv[]){

  if(argc<3){
    std::cerr<<"ERROR, I need some parameters!!\nTry to run me with:\n./a.out 100000 5000 0.01"<<std::endl;
    exit(1);
  }
  //Number of particles and box size set to match dens=4 as in [1]
  int N = 16384;

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
  Box box(16);
  //Initial positions
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto vel = pd->getVel(access::location::cpu, access::mode::write);    

    auto initial =  initLattice(box.boxSize, N, fcc);
    
    //Start in a cubic lattice, pos.w contains the particle type
    //auto initial = cubicLattice(box.boxSize, N);
    
    fori(0,N){
      pos.raw()[i] = make_real4(sys->rng().uniform3(-box.boxSize.x*0.5,box.boxSize.x*0.5), 0);//initial[i];
      //Type of particle is stored in .w
      pos.raw()[i].w = 0;
      vel.raw()[i] = make_real3(sys->rng().gaussian(0, 1),sys->rng().gaussian(0, 1),sys->rng().gaussian(0, 1));
    }    
  }
  

  //Modules can work on a certain subset of particles if needed, the particles can be grouped following any criteria
  //The builtin ones will generally work faster than a custom one. See ParticleGroup.cuh for a list
  
  //A group created with no criteria will contain all the particles  
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  
  ofstream out("kk");
  
  using NVE = VerletNVE;
  
  NVE::Parameters par;

  par.dt = std::stod(argv[3]);
  par.initVelocities = false;

  auto verlet = make_shared<NVE>(pd, pg, sys, par);

  using PairForces = PairForces<Potential::DPD>;
  
  //This is the general interface for setting up a potential  
  Potential::DPD::Parameters dpd_params;
  //Set too match parameters in [1]
  dpd_params.cutOff = 1.0; 
  dpd_params.temperature = 1.0;
  dpd_params.gamma = 4.0;
  dpd_params.A = 25.0;
  dpd_params.dt = par.dt;
  
  auto pot = make_shared<Potential::DPD>(sys, dpd_params);

  PairForces::Parameters params;
  params.box = box;  //Box to work on
  auto pairforces = make_shared<PairForces>(pd, pg, sys, params, pot);

  verlet->addInteractor(pairforces);

  sys->log<System::MESSAGE>("RUNNING!!!");

  pd->sortParticles();
        
  Timer tim;
  tim.tic();
  int nsteps = std::atoi(argv[1]);
  int printSteps = std::atoi(argv[2]);
  //Thermalization
  forj(0,1000){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    verlet->forwardTime();
  }
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
	out<<p<<" "<<0.5<<" "<<type<<endl;
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
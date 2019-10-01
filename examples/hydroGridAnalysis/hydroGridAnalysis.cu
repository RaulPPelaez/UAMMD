/*Raul P. Pelaez 2017. An example of how to use HydroGrid in UAMMD.

  See line 145 for HG setup.

  This is a simulation of a 2D LJ liquid with Brownian Dynamics.
  HG is called every printSteps to update according to hydroGridOptions.nml (which must be present at the current folder) and write to disk.

  Some parameters are set in the command line:
  Try this for a quick test: ./a.out 15 256 0.0001 1000000 1000 0.5
  You can see a description below

  You can visualize the particle results with superpunto. HG will output whatever was asked in the .nml.

  See utils/HydroGrid.cuh for more information:

References:
[1] https://github.com/stochasticHydroTools/HydroGrid

*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/BrownianDynamics.cuh"
#include"Interactor/PairForces.cuh"
// #include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"utils/HydroGrid.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>


using namespace uammd;
using namespace std;


int main(int argc, char *argv[]){
  if(argc != 7){
    cerr<<"ERROR: Not the right arguments!, try to run me with: ./a.out 15 256 0.0001 1000000 1000 0.5"<<endl;
    cerr<<"N=pow(w, argv[1])"<<endl;
    cerr<<"L=argv[2]"<<endl;
    cerr<<"dt=argv[3]"<<endl;
    cerr<<"nsteps=argv[4]"<<endl;
    cerr<<"printSteps=argv[5]"<<endl;
    cerr<<"Temperature=argv[6]"<<endl;

    exit(1);
  }

  int N = pow(2,atoi(argv[1]));//atoi(argv[1]));

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
  real L = std::stod(argv[2]);
  Box box(make_real3(L,L,0));
  //Initial positions
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    //auto radius = pd->getRadius(access::location::cpu, access::mode::write);

    //Start in a square lattice, pos.w contains the particle type
    auto initial =  initLattice(box.boxSize, N, sq);

    fori(0,N){
      pos.raw()[i] = initial[i];
      //Type of particle is stored in .w
      pos.raw()[i].w = sys->rng().next()%3;
    }

  }

  //Modules can work on a certain subset of particles if needed, the particles can be grouped following any criteria
  //The builtin ones will generally work faster than a custom one. See ParticleGroup.cuh for a list

  //A group created with no criteria will contain all the particles
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");


  BD::EulerMaruyama::Parameters par;
  par.temperature = std::stod(argv[6]);
  par.viscosity = 1.0/(6*M_PI);
  par.hydrodynamicRadius = 1.0;
  par.dt = std::stod(argv[3]);
  par.is2D=true;

  auto bd = make_shared<BD::EulerMaruyama>(pd, pg, sys, par);

   using PairForces = PairForces<Potential::LJ>;

   //This is the general interface for setting up a potential
   auto pot = make_shared<Potential::LJ>(sys);
   {
     //Each Potential describes the pair interactions with certain parameters.
     //The needed ones are in InputPairParameters inside each potential, in this case:
     Potential::LJ::InputPairParameters par;
     par.epsilon = 1.0;
     par.shift = false;

     par.sigma = 1.0;
     par.cutOff = 2.5*par.sigma;
     pot->setPotParameters(0, 0, par);
   }

   PairForces::Parameters params;
   params.box = box;  //Box to work on
   auto pairforces = make_shared<PairForces>(pd, pg, sys, params, pot);

   bd->addInteractor(pairforces);

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

  ofstream out("particle.pos"); //File to output particle positions

  //HydroGrid configuration
  HydroGrid::Parameters hgpar;
  hgpar.box = box;                           //Simulation box
  hgpar.cellDim = make_int3(128, 128, 1);    //cells to perform HG analysis
  hgpar.dt = par.dt;                         //Time between steps
  hgpar.outputName = "run";                  //Name prefix of HG output
  hgpar.useColors = true;                    //Use pos.w as HydroGrid species
  HydroGrid hg(pd, sys, hgpar);


  hg.init(); //Initialize

  //Run the simulation
  forj(0,nsteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    bd->forwardTime();

    //Write results
    if(j%printSteps==0 && printSteps > 0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");

      //Update HG and write current results
      hg.update(j);
      hg.write(j);

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
	out<<p<<" "<<0.5<<" "<<type<<"\n";
      }
      out<<flush;
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
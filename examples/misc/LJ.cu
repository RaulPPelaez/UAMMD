/*Raul P. Pelaez 2019-2021. A short range forces example.

It describes a LJ liquid simulation in a periodic box.
Two types of LJ particles exist, starting in a random configuration.
Each type is attracted to a different z plane.
This makes the particles form a kind of pillars going from one type to the other.

Commented code is available to show other available features.

You can visualize the reuslts with superpunto

 */

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/VerletNVT.cuh"
#include"Interactor/ExternalForces.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/NeighbourList/VerletList.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include<fstream>
#include <random>

using namespace uammd;
using std::make_shared;
using std::endl;

//An HarmonicWall functor to be used in a ExternalForces module (See ExternalForces.cuh)
struct HarmonicWall: public ParameterUpdatable{
  real zwall;
  real k = 0.1;
  HarmonicWall(real zwall):zwall(zwall){}

  __device__ ForceEnergyVirial sum(Interactor::Computables comp, real4 pos){
    return {{0.0f, 0.0f, -k*(pos.z-zwall)}, 0, 0};
  }

  //If this function is not present, energy is assumed to be zero
  // __device__real energy(real4 pos){

  //   return real(0.5)*k*pow(pos.z-zwall, 2);
  // }

  //This function must return a list of arrays whose elements force/energy will need
  auto getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::gpu, access::read);
    return pos.begin();
    //If there are more arrays:
    //return std::make_tuple(pos.begin(), otherthing1.begin(), otherthing2.begin());
  }

  //void updateSimulationTime(real time){
    //You can be aware of changes in some parameters by making the functor ParameterUpdatable
    //and overriding the update function you want, see misc/ParameterUpdatable.h for a list
  //}
};

struct Parameters{
  real3 boxSize;
  real dt;
  std::string outputFile;
  int numberParticles;
  int numberSteps, printSteps;
  real temperature, friction;
};

//Let us group the UAMMD simulation in this struct that we can pass around
struct UAMMD{
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
  Parameters par;
};

Parameters readParameters(std::string file);

//Initialize the basic UAMMD structures and read parameters from the data.main.lj file
UAMMD initializeUAMMD(int argc, char *argv[]){
  UAMMD sim;
  //UAMMD System entity holds information about the GPU and tools to interact with the computer itself (such as a loging system). All modules need a System to work on.
  sim.sys = make_shared<System>(argc, argv);
  sim.par = readParameters("data.main.lj");
  //Modules will ask System when they need a random number (i.e for seeding the GPU RNG).
  auto seed = std::random_device()();
  sim.sys->rng().setSeed(seed);
  //ParticleData stores all particle properties the simulation will need.
  //Needs to start with a certain number of particles
  sim.pd = make_shared<ParticleData>(sim.par.numberParticles, sim.sys);
  return sim;
}
//Set the initial positons and types for the particles
void initializeParticles(UAMMD sim){
  //Ask pd for a property like so:
  auto pos = sim.pd->getPos(access::cpu, access::write);
  auto initial =  initLattice(sim.par.boxSize, sim.par.numberParticles, fcc);
  //Start in an fcc lattice, pos.w contains the particle type, lets make it ranfomly 0 or 1
  std::transform(initial.begin(), initial.end(), pos.begin(),
		 [&](real4 p){
		   p.w = sim.sys->rng().uniform(0,1) > 0.5?0:1;
		   return p;
		 });
}

std::shared_ptr<Integrator> createIntegrator(UAMMD sim){
  //Some modules need additional parameters, in this case VerletNVT needs dt, temperature...
  //When additional parameters are needed, they need to be supplied in a form similar to this:
  using NVT = VerletNVT::GronbechJensen;
  NVT::Parameters par;
  par.temperature = sim.par.temperature;
  par.dt = sim.par.dt;
  par.friction = sim.par.friction;
  auto verlet = std::make_shared<NVT>(sim.pd, par);
  return verlet;
}

void addExternalPotentialInteractions(std::shared_ptr<Integrator> verlet, UAMMD sim){
  //Modules can work on a certain subset of particles if needed, the particles can be grouped following any criteria
  //The builtin ones will generally work faster than a custom one. See ParticleGroup.cuh for a list
  //A group created with no criteria will contain all the particles
  auto pg = make_shared<ParticleGroup>(sim.pd, "All");
  auto pg2 = make_shared<ParticleGroup>(particle_selector::Type(0), sim.pd, "12");
  auto pg3 = make_shared<ParticleGroup>(particle_selector::Type(1), sim.pd, "11");
  //Harmonic walls acting on different particle groups
  //This two interactors will cause particles in group pg2 to stick to a wall in -Lz/4
  //And the ones in pg3 to +Lz/4
  real Lz = sim.par.boxSize.z;
  auto extForces = make_shared<ExternalForces<HarmonicWall>>(pg2,
							     make_shared<HarmonicWall>(-Lz*0.25));
  auto extForces2 = make_shared<ExternalForces<HarmonicWall>>(pg3,
							      make_shared<HarmonicWall>(Lz*0.25));
  //Add interactors to integrator.
  verlet->addInteractor(extForces);
  verlet->addInteractor(extForces2);
}

std::shared_ptr<Potential::LJ> createLJPotential(UAMMD sim){
  //This is the general interface for setting up a potential
  auto pot = make_shared<Potential::LJ>();
  //Each Potential describes the pair interactions with certain parameters.
  //The needed ones are in InputPairParameters inside each potential, in this case:
  Potential::LJ::InputPairParameters par;
  par.epsilon = 1.0;
  par.shift = false;
  par.sigma = 2;
  par.cutOff = 2.5*par.sigma;
  //Once the InputPairParameters has been filled accordingly for a given pair of types,
  //a potential can be informed like this:
  pot->setPotParameters(1, 1, par);
  par.sigma = 1.0;
  par.cutOff = 2.5*par.sigma;
  pot->setPotParameters(0, 0, par);
  par.epsilon = 4.0;
  par.sigma = 0.5*(2.0+1.0);
  par.cutOff = 2.5*par.sigma;
  //the pair 1,0 is registered as well with this call, and assumed to be the same
  pot->setPotParameters(0, 1, par);
  return pot;
}

void addShortRangeInteraction(std::shared_ptr<Integrator> verlet, UAMMD sim){
  //You can force the use of a certain neighbour list passing its name as a second template argument
  //The default list is a CellList
  using PairForces = PairForces<Potential::LJ, VerletList>;
  //Modules working with pairs of particles usually ask for a Potential object
  auto pot = createLJPotential(sim);
  PairForces::Parameters params;
  //Some modules need a simulation box (i.e PairForces for the PBC)
  params.box = Box(sim.par.boxSize);  //Box to work on
  auto pairforces = make_shared<PairForces>(sim.pd, params, pot);
  verlet->addInteractor(pairforces);
}


void printSimulation(UAMMD sim, std::ofstream &out){
  sim.sys->log<System::DEBUG1>("[System] Writing to disk...");
  auto pos = sim.pd->getPos(access::cpu, access::read);
  //The order of ParticleData::get* arrays might change,
  //Accesing the arrays using this indices will ensure the traversing order is always the initial one
  const int * index2id = sim.pd->getIdOrderedIndices(access::location::cpu);
  Box box(sim.par.boxSize);
  out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
  std::for_each(index2id, index2id + pos.size(),
		[&](int id){
		  real4 pc = pos[id];
		  real3 p = box.apply_pbc(make_real3(pc));
		  int type = pc.w;
		  out<<p<<" "<<0.5*(type==1?2:1)*pow(2,1/6.)<<" "<<type<<"\n";
		});
}

void runSimulation(std::shared_ptr<Integrator> verlet, UAMMD sim){
  std::ofstream out(sim.par.outputFile);
  //Run the simulation
  forj(0, sim.par.numberSteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    verlet->forwardTime();
    if(sim.par.printSteps>0 and j%sim.par.printSteps==0){
      printSimulation(sim, out);
    }
    //You can ask ParticleData to sort the particles every few steps
    //It is not an expensive thing to do really and can increase performance
    if(j%500 == 0){
      sim.pd->sortParticles();
    }
  }
}

int main(int argc, char *argv[]){
  UAMMD sim = initializeUAMMD(argc, argv);
  initializeParticles(sim);
  auto verlet = createIntegrator(sim);
  addExternalPotentialInteractions(verlet, sim);
  addShortRangeInteraction(verlet, sim);
  //You can issue a logging event like this, many log levels exists (see System.cuh).
  //A maximum log level is set in System.cuh or through the MAXLOGLEVEL macro
  //every logging event with a level superior to the max will be ignored by the compiler
  //so dont be afraid to write System::DEBUGX log calls.
  sim.sys->log<System::MESSAGE>("RUNNING!!!");
  Timer tim; tim.tic();
  runSimulation(verlet, sim);
  auto totalTime = tim.toc();
  sim.sys->log<System::MESSAGE>("mean FPS: %.2f", sim.par.numberSteps/totalTime);
  //sys->finish() will ensure a smooth termination of any UAMMD module.
  sim.sys->finish();
  return 0;
}

Parameters readParameters(std::string file){
  {
    if(!std::ifstream(file).good()){
      real visco = 1/(6*M_PI);
      std::ofstream default_options(file);
      default_options<<"boxSize 90 90 90"<<std::endl;
      default_options<<"numberParticles 16384"<<std::endl;
      default_options<<"dt 0.01"<<std::endl;
      default_options<<"numberSteps 20000"<<std::endl;
      default_options<<"printSteps 100"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
      default_options<<"temperature 0.1"<<std::endl;
      default_options<<"friction "<<visco<<std::endl;
    }
  }
  Parameters par;
  InputFile in(file);
  in.getOption("boxSize", InputFile::Required)>>par.boxSize.x>>par.boxSize.y>>par.boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>par.numberSteps;
  in.getOption("printSteps", InputFile::Required)>>par.printSteps;
  in.getOption("dt", InputFile::Required)>>par.dt;
  in.getOption("numberParticles", InputFile::Required)>>par.numberParticles;
  in.getOption("outputFile", InputFile::Required)>>par.outputFile;
  in.getOption("temperature", InputFile::Required)>>par.temperature;
  in.getOption("friction", InputFile::Required)>>par.friction;
  return par;
}

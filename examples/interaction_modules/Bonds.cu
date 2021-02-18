/*Raul P. Pelaez 2019-2021. A bonded forces usage example

  A group of dumbells interacting inside a viscous fluid (integrated with brownian dynamics). Dumbells are formed by two particles with two different types, each particle is only attracted to particles of the same type (and are repulsive otherwise).
  Furthemore some dumbells are attracted to one wall by a gravity like potential, while the rest are attracted to the opossite wall.
  This results in beautiful pillars with stripe like patterns.

Reads some parameters from a file called "data.main.bonds", if not present it will be auto generated with some default parameters.


You can visualize the reuslts with superpunto

*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/BrownianDynamics.cuh"
#include"Interactor/BondedForces.cuh"
#include"Interactor/PairForces.cuh"
#include"Interactor/Potential/Potential.cuh"
#include"Interactor/ExternalForces.cuh"
#include"utils/InitialConditions.cuh"
#include"utils/InputFile.h"
#include<fstream>
using namespace uammd;
//The particles fall due to a gravity like force until they reach a wall.
//An Wall+Gravity functor to be used in a ExternalForces module (See ExternalForces.cuh)
struct Wall{
  real k = 20;
  real3 L;
  Wall(real3 L):L(L){
  }

  __device__ real3 force(real4 pos, int id){
    real3 f = real3();
    if(pos.x<-L.x*0.5f) f.x = k;  if(pos.x>L.x*0.5f) f.x = -k;
    if(pos.y<-L.y*0.5f) f.y = k;  if(pos.y>L.y*0.5f) f.y = -k;
    if(pos.z<-L.z*0.5f) f.z = k;  if(pos.z>L.z*0.5f) f.z = -k;
    if((id/2)%2 == 0)
      f.z += k/40.0f;
    else
      f.z -= k/40.0f;
    return f;
  }

  //If this function is not present, energy is assumed to be zero
  // __device__ real energy(real4 pos){
  //   return real(0.5)*k*pow(pos.z-zwall, 2);
  // }

  auto getArrays(ParticleData* pd){
    auto pos = pd->getPos(access::gpu, access::read);
    auto id = pd->getId(access::gpu, access::read);
    return std::make_tuple(pos.begin(), id.begin());
  }

};

using std::make_shared;
using std::endl;

struct Parameters{
  real3 boxSize;
  real dt;
  std::string outputFile;
  int numberParticles;
  int numberSteps, printSteps;
  real temperature, viscosity;
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
  sim.par = readParameters("data.main.bonds");
  //Let us have only an even number of particles
  if(sim.par.numberParticles%2 != 0) sim.par.numberParticles++;
  //Modules will ask System when they need a random number (i.e for seeding the GPU RNG).
  auto seed = std::random_device()();
  sim.sys->rng().setSeed(seed);
  //ParticleData stores all particle properties the simulation will need.
  //Needs to start with a certain number of particles
  sim.pd = make_shared<ParticleData>(sim.par.numberParticles, sim.sys);
  return sim;
}
//Set the initial positons for the particles, also write the bond file
void initializeParticlesAndBonds(UAMMD sim){
  //Ask pd for a property like so:
  auto pos = sim.pd->getPos(access::cpu, access::write);
  //Start half the particles in a fcc lattice
  auto initial =  initLattice(sim.par.boxSize*0.8, sim.par.numberParticles/2, fcc);
  std::ofstream out("bonds.dat");
  out<<sim.par.numberParticles/2<<std::endl;
  //replicate each particle to the right and place a bond between both.
  fori(0, sim.par.numberParticles/2){
    pos[2*i] = initial[i];
    auto bonded = initial[i];
    bonded.x += 1.25;
    bonded.w = 1;
    pos[2*i+1] = bonded;
    out<<2*i<<" "<<2*i+1<<" 200 1.25"<<"\n";
  }

}

std::shared_ptr<Integrator> createIntegrator(UAMMD sim){
  //Some modules need additional parameters, in this case BD needs dt, temperature...
  //When additional parameters are needed, they need to be supplied in a form similar to this:
  using BD = BD::EulerMaruyama;
  BD::Parameters par;
  par.temperature = sim.par.temperature;
  par.dt = sim.par.dt;
  par.hydrodynamicRadius = 1.0;
  par.viscosity = sim.par.viscosity;
  auto bd = std::make_shared<BD>(sim.pd, sim.sys, par);
  return bd;
}

void addHarmonicBondInteraction(std::shared_ptr<Integrator> bd, UAMMD sim){
  //Let us use Harmonic bonds
  using BondedForces = BondedForces<BondedType::Harmonic>;
  BondedForces::Parameters params;
  params.file = "bonds.dat";  //We wrote this file in initializeParticles
  auto bondedforces = make_shared<BondedForces>(sim.pd, sim.sys, params);
  bd->addInteractor(bondedforces);
}

void addExternalPotentialInteractions(std::shared_ptr<Integrator> bd, UAMMD sim){
  auto externalForces = make_shared<ExternalForces<Wall>>(sim.pd, sim.sys, make_shared<Wall>(sim.par.boxSize*0.9));
  bd->addInteractor(externalForces); 
}

std::shared_ptr<Potential::LJ> createLJPotential(UAMMD sim){
  //This is the general interface for setting up a potential
  auto pot = make_shared<Potential::LJ>(sim.sys);
  //Each Potential describes the pair interactions with certain parameters.
  //The needed ones are in InputPairParameters inside each potential, in this case:
  Potential::LJ::InputPairParameters par;
  par.epsilon = 1.0;
  par.shift = false;
  par.sigma = 1;
  par.cutOff = 2.5*par.sigma;
  pot->setPotParameters(0, 0, par);
  pot->setPotParameters(1, 1, par);
  par.cutOff = par.sigma*pow(2, 1/6.0);
  pot->setPotParameters(0, 1, par);
  return pot;
}

void addShortRangeInteraction(std::shared_ptr<Integrator> bd, UAMMD sim){
  using PairForces = PairForces<Potential::LJ>;
  //Modules working with pairs of particles usually ask for a Potential object
  auto pot = createLJPotential(sim);
  PairForces::Parameters params;
  //Some modules need a simulation box (i.e PairForces for the PBC)
  params.box = Box(sim.par.boxSize);  //Box to work on
  auto pairforces = make_shared<PairForces>(sim.pd, sim.sys, params, pot);
  bd->addInteractor(pairforces);
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
		  out<<p<<" "<<0.5*pow(2,1/6.)<<" "<<type<<"\n";
		});
}

void runSimulation(std::shared_ptr<Integrator> bd, UAMMD sim){
  std::ofstream out(sim.par.outputFile);
  //Run the simulation
  forj(0, sim.par.numberSteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    bd->forwardTime();
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
  initializeParticlesAndBonds(sim);
  auto bd = createIntegrator(sim);
  addExternalPotentialInteractions(bd, sim);
  addHarmonicBondInteraction(bd, sim);
  addShortRangeInteraction(bd, sim);
  //You can issue a logging event like this, many log levels exists (see System.cuh).
  //A maximum log level is set in System.cuh or through the MAXLOGLEVEL macro
  //every logging event with a level superior to the max will be ignored by the compiler
  //so dont be afraid to write System::DEBUGX log calls.
  sim.sys->log<System::MESSAGE>("RUNNING!!!");
  Timer tim; tim.tic();
  runSimulation(bd, sim);
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
      default_options<<"boxSize 45 45 45"<<std::endl;
      default_options<<"numberParticles 8000"<<std::endl;
      default_options<<"dt 0.00075"<<std::endl;
      default_options<<"numberSteps 1000000"<<std::endl;
      default_options<<"printSteps 1000"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
      default_options<<"temperature 0.2"<<std::endl;
      default_options<<"viscosity "<<visco<<std::endl;
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
  in.getOption("viscosity", InputFile::Required)>>par.viscosity;
  return par;
}

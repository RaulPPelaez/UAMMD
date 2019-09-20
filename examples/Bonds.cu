/*Raul P. Pelaez 2019. A bonded forces usage example

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

  __device__ __forceinline__ real3 force(const real4 &pos, int id){

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
  // __device__ __forceinline__ real energy(const real4 &pos){

  //   return real(0.5)*k*pow(pos.z-zwall, 2);
  // }

  std::tuple<const real4 *, const int *> getArrays(ParticleData* pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto id = pd->getId(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw(), id.raw());
  }


};

using std::make_shared;
using std::endl;

real3 boxSize;
real dt;
std::string outputFile;
int numberParticles;
int numberSteps, printSteps;
real temperature, viscosity;
void readParameters(std::shared_ptr<System> sys, std::string file);

int main(int argc, char *argv[]){


  //UAMMD System entity holds information about the GPU and tools to interact with the computer itself (such as a loging system). All modules need a System to work on.

  auto sys = make_shared<System>(argc, argv);

  readParameters(sys, "data.main.bonds");
    //Let us have only an even number of particles
  if(numberParticles%2 != 0) numberParticles++;

  //Modules will ask System when they need a random number (i.e for seeding the GPU RNG).
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  //ParticleData stores all the needed properties the simulation will need.
  //Needs to start with a certain number of particles, which can be changed mid-simulation
  //If UAMMD is to be used as a plugin for other enviroment or custom code, ParticleData should accept references to
  // properties allocated and handled by the user, this is a non-implemented work in progress as of now though.
  auto pd = make_shared<ParticleData>(numberParticles, sys);

  //Some modules need a simulation box (i.e PairForces for the PBC)
  Box box(boxSize);
  //Initial positions
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);

    //Start half the particles in a fcc lattice
    auto initial =  initLattice(box.boxSize*0.8, numberParticles/2, fcc);
    std::ofstream out("bonds.dat");
    out<<numberParticles/2<<std::endl;
    //replicate each particle to the right and place a bond between the both.
    fori(0, numberParticles/2){
      pos[2*i] = initial[i];
      auto bonded = initial[i];
      bonded.x += 1.25;
      bonded.w = 1;
      pos[2*i+1] = bonded;
      out<<2*i<<" "<<2*i+1<<" 200 1.25"<<"\n";
    }


  }



  //Initialize a Brownian Dynamics integrator
  BD::EulerMaruyama::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = 1.0;
  par.dt = dt;

  auto bd = make_shared<BD::EulerMaruyama>(pd, sys, par);

  {
    //Let us use Harmonic bonds
    using BondedForces = BondedForces<BondedType::Harmonic>;

    //You can use Elastic_Network_Model.cpp to generate some example bonds for the starting configuration.
    BondedForces::Parameters params;

    params.file = "bonds.dat";  //Box to work on

    auto bondedforces = make_shared<BondedForces>(pd, sys, params);

    bd->addInteractor(bondedforces);
  }
  {
    auto externalForces = make_shared<ExternalForces<Wall>>(pd, sys,
							    make_shared<Wall>(box.boxSize*0.9));
    bd->addInteractor(externalForces);
  }

    {
    using PairForces = PairForces<Potential::LJ>;
    auto pot = make_shared<Potential::LJ>(sys);
    //Type 0 particles are attractive, type 1 are repulsive
    {
      Potential::LJ::InputPairParameters par;
      par.epsilon = 1.0;
      par.shift = false;

      par.sigma = 1;
      par.cutOff = 2.5*par.sigma;

      pot->setPotParameters(0, 0, par);
      pot->setPotParameters(1, 1, par);

      par.cutOff = par.sigma*pow(2, 1/6.0);
      pot->setPotParameters(0, 1, par);
    }

    PairForces::Parameters params;
    params.box = box;  //Box to work on
    auto pairforces = make_shared<PairForces>(pd, sys, params, pot);

    bd->addInteractor(pairforces);
  }



  //You can issue a logging event like this, a wide variety of log levels exists (see System.cuh).
  //A maximum log level is set in System.cuh, every logging event with a level superior to the max will result in
  // absolutely no overhead, so dont be afraid to write System::DEBUGX log calls.
  sys->log<System::MESSAGE>("RUNNING!!!");
  std::ofstream out(outputFile);
  //Ask ParticleData to sort the particles in memory!
  //It is a good idea to sort the particles once in a while during the simulation
  //This can increase performance considerably as it improves coalescence.
  //Sorting the particles will cause the particle arrays to change in order and (possibly) address.
  //This changes will be informed with signals and any module that needs to be aware of such changes
  //will acknowedge it through a callback (see ParticleData.cuh).
  pd->sortParticles();

  Timer tim;
  tim.tic();
  //Run the simulation
  forj(0, numberSteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    bd->forwardTime();

    //Write results
    if(printSteps>0 and j%printSteps==0){
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);

      //This allows to access the particles with the starting order so the particles are written in the same order
      // even after a sorting
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);

      out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
      real3 p;
      fori(0, numberParticles){
	real4 pc = pos[sortedIndex[i]];
	p = make_real3(pc);
	int type = pc.w;
	out<<p<<" 0.5 "<<type*5<<"\n";
      }
    }
    //Sort the particles every few steps
    //It is not an expensive thing to do really.
    if(j%1000 == 0){
      pd->sortParticles();
    }
  }

  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", numberSteps/totalTime);
  //sys->finish() will ensure a smooth termination of any UAMMD module.
  sys->finish();

  return 0;
}

void readParameters(std::shared_ptr<System> sys, std::string file){

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

  InputFile in(file, sys);

  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("numberParticles", InputFile::Required)>>numberParticles;
  in.getOption("outputFile", InputFile::Required)>>outputFile;
  in.getOption("temperature", InputFile::Required)>>temperature;
  in.getOption("viscosity", InputFile::Required)>>viscosity;
}
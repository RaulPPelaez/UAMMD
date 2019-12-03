/*Raul P. Pelaez 2019. A short range forces example.

Describes a LJ liquid simulation in a periodic box.

Reads some parameters from a file called "data.main.benchmark", if not present it will be auto generated with some default parameters.

You should get ~ 90 FPS on a GTX 980

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
#include"utils/InputFile.h"
#include<fstream>


using namespace uammd;
using std::make_shared;
using std::endl;

real3 boxSize;
real dt;
std::string outputFile;
int numberParticles;
int numberSteps, printSteps;
real temperature, viscosity;
void readParameters(std::shared_ptr<System> sys, std::string file);
bool periodicityX, periodicityY, periodicityZ;
int main(int argc, char *argv[]){

  //UAMMD System entity holds information about the GPU and tools to interact with the computer itself (such as a loging system). All modules need a System to work on.

  auto sys = make_shared<System>(argc, argv);
  readParameters(sys, "data.main.benchmark");

  //Modules will ask System when they need a random number (i.e for seeding the GPU RNG).
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  //ParticleData stores all the needed properties the simulation will need.
  //Needs to start with a certain number of particles, which can be changed mid-simulation
  auto pd = make_shared<ParticleData>(numberParticles, sys);

  //Some modules need a simulation box (i.e PairForces for the PBC)
  Box box(boxSize);
  box.setPeriodicity(periodicityX, periodicityY, periodicityZ);
  //Initial positions
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto initial =  initLattice(box.boxSize, numberParticles, fcc);
    std::copy(initial.begin(), initial.end(), pos.begin());
  }

  std::ofstream out(outputFile);


  using NVT = VerletNVT::GronbechJensen;
  NVT::Parameters par;
  par.temperature = temperature;
  par.dt = dt;
  par.viscosity = viscosity;
  auto verlet = make_shared<NVT>(pd, sys, par);

  {
    //Modules working with pairs of particles usually ask for a Potential object
    //PairForces decides if it should use a neighbour list or treat the system as NBody,
    //You can force the use of a certain neighbour list passing its name as a second template argument

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
      //Once the InputPairParameters has been filled accordingly for a given pair of types,
      //a potential can be informed like this:
      pot->setPotParameters(0, 0, par);
    }

    PairForces::Parameters params;
    params.box = box;  //Box to work on
    auto pairforces = make_shared<PairForces>(pd, sys, params, pot);
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
  //Run the simulation
  forj(0, numberSteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    verlet->forwardTime();

    //Write results
    if(printSteps > 0 and j%printSteps==1)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      //continue;
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;

      fori(0, numberParticles){
	real4 pc = pos[sortedIndex[i]];
	real3 p = box.apply_pbc(make_real3(pc));
	out<<p<<" "<<0.5<<" "<<0<<"\n";
      }

    }
    //Sort the particles every few steps
    //It is not an expensive thing to do really.
    if(j%500 == 0){
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
      std::ofstream default_options(file);
      default_options<<"boxSize 128 128 128"<<std::endl;
      default_options<<"numberParticles 1048576"<<std::endl;
      default_options<<"dt 0.01"<<std::endl;
      default_options<<"numberSteps 500"<<std::endl;
      default_options<<"printSteps -1"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
      default_options<<"temperature 1.0"<<std::endl;
      default_options<<"viscosity 1"<<std::endl;
      default_options<<"periodicity 1 1 1"<<std::endl;
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
  in.getOption("periodicity", InputFile::Required)>>periodicityX>>periodicityY>>periodicityZ;
}
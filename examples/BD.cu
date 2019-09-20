/*Raul P. Pelaez 2019. A Brownian Dynamics example.

Runs a Brownian Dynamics simulation with particles starting in a box. Particles have two different radius

You can visualize the results with superpunto

Reads some parameters from a file called "data.main.bd", if not present it will be auto generated with some default parameters.

Test for correctness:
With the default parameters the diffussion coefficient of each particle should be just 1. The diffussion coefficient can be computed through the slope of the mean square displacement, as MSD(t) = 2*D*t in each direction. This script computes the diffussion coefficient this way, provided a msd program*:

./bd | fastmsd -N 16384 -Nsteps 200 | awk '{print $1*0.005*100, ($2+$3+$4)/(6.0)}' | awk '{x[NR] = $1; y[NR] = $2; sx += x[NR]; sy += y[NR]; sxx += x[NR]*x[NR]; sxy += x[NR]*y[NR];}END{det = NR*sxx - sx*sx; a = (NR*sxy - sx*sy)/det; b = (-sx*sxy+sxx*sy)/det; print a, b;}'

This line should output the diffusion coefficient, 1 and the value at t=0 of the MSD, which should be zero.

*You can use https://github.com/RaulPPelaez/MeanSquareDisplacement

*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
//The rest can be included depending on the used modules
#include"Integrator/BrownianDynamics.cuh"

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

int main(int argc, char *argv[]){
  //UAMMD System entity holds information about the GPU and tools to interact with the computer itself (such as a loging system). All modules need a System to work on.

  auto sys = make_shared<System>(argc, argv);
  readParameters(sys, "data.main.bd");
  //Modules will ask System when they need a random number (i.e for seeding the GPU RNG).
  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  //ParticleData stores all the needed properties the simulation will need.
  //Needs to start with a certain number of particles, which can be changed mid-simulation
  auto pd = make_shared<ParticleData>(numberParticles, sys);

  //Some modules need a simulation box (i.e PairForces for the PBC)
  Box box(boxSize);
  //Initial positions
  {
    //Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    //auto radius = pd->getRadius(access::location::cpu, access::mode::write);

    //Generate some positions in a fcc lattice
    auto initial =  initLattice(box.boxSize, numberParticles, fcc);

    std::copy(initial.begin(), initial.end(), pos.begin());
  }

  //Initialize the integrator module
  BD::EulerMaruyama::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.hydrodynamicRadius = 1.0;
  par.dt = dt;
  auto bd = make_shared<BD::EulerMaruyama>(pd, sys, par);

  //You can issue a logging event like this, a wide variety of log levels exists (see System.cuh).
  //A maximum log level is set in System.cuh, every logging event with a level superior to the max will result in
  // absolutely no overhead, so dont be afraid to write System::DEBUGX log calls.
  sys->log<System::MESSAGE>("RUNNING!!!");

  std::ofstream out(outputFile);
  Timer tim;
  tim.tic();
  //Run the simulation
  forj(0,numberSteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    bd->forwardTime();

    //Write results
    if(printSteps and j%printSteps==0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      //auto radius = pd->getRadius(access::location::cpu, access::mode::read);
      //This allows to access the particles with the starting order so the particles are written in the same order
      // even after a sorting
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);

      out<<"#"<<endl;
      real3 p;
      fori(0, numberParticles){
	real4 pc = pos.raw()[sortedIndex[i]];
	p = make_real3(pc);
	int type = pc.w;
	out<<p<<" "<<0.5<<" "<<type<<"\n";
      }
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
      default_options<<"boxSize 25 25 25"<<std::endl;
      default_options<<"numberParticles 16384"<<std::endl;
      default_options<<"dt 0.005"<<std::endl;
      default_options<<"numberSteps 20000"<<std::endl;
      default_options<<"printSteps 100"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
      default_options<<"temperature 1.0"<<std::endl;
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
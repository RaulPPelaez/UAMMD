/*Raul P. Pelaez 2019. A short range forces example.

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
#include"Interactor/NBodyForces.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/NeighbourList/VerletList.cuh"
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

//An HarmonicWall functor to be used in a ExternalForces module (See ExternalForces.cuh)
struct HarmonicWall: public ParameterUpdatable{
  real zwall;
  real k = 0.1;
  HarmonicWall(real zwall):zwall(zwall){}

  __device__ __forceinline__ real3 force(const real4 &pos){
    return make_real3(0.0f, 0.0f, -k*(pos.z-zwall));
  }

  //If this function is not present, energy is assumed to be zero
  // __device__ __forceinline__ real energy(const real4 &pos){

  //   return real(0.5)*k*pow(pos.z-zwall, 2);
  // }

  std::tuple<const real4 *> getArrays(ParticleData *pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }

  void updateSimulationTime(real time){
    //You can be aware of changes in some parameters by making the functor ParameterUpdatable
    //and overriding the update function you want, see misc/ParameterUpdatable.h for a list
  }
};

int main(int argc, char *argv[]){

  //UAMMD System entity holds information about the GPU and tools to interact with the computer itself (such as a loging system). All modules need a System to work on.

  auto sys = make_shared<System>(argc, argv);
  readParameters(sys, "data.main.lj");
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
    auto initial =  initLattice(box.boxSize, numberParticles, fcc);

    //Start in a cubic lattice, pos.w contains the particle type
    //auto initial = cubicLattice(box.boxSize, N);
    std::transform(initial.begin(), initial.end(), pos.begin(),
		   [&](real4 p){
		     p.w = sys->rng().uniform(0,1) > 0.5?0:1;
		     return p;
		   });
  }


  //Modules can work on a certain subset of particles if needed, the particles can be grouped following any criteria
  //The builtin ones will generally work faster than a custom one. See ParticleGroup.cuh for a list

  //A group created with no criteria will contain all the particles
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  auto pg2 = make_shared<ParticleGroup>(particle_selector::Type(0), pd, sys, "12");
  auto pg3 = make_shared<ParticleGroup>(particle_selector::Type(1), pd, sys, "11");
  // auto pg4 = make_shared<ParticleGroup>(particle_selector::Type(2), pd, sys, "10");


  //Some modules need additional parameters, in this case VerletNVT needs dt, temperature...
  //When additional parameters are needed, they need to be supplied in a form similar to this:
  using NVT = VerletNVT::GronbechJensen;

  NVT::Parameters par;
  par.temperature = temperature;
  par.dt = dt;
  par.viscosity = viscosity;
  auto verlet = make_shared<NVT>(pd, pg, sys, par);

  //Harmonic walls acting on different particle groups
  //This two interactors will cause particles in group pg2 to stick to a wall in -Lz/4
  //And the ones in pg3 to +Lz/4
  auto extForces = make_shared<ExternalForces<HarmonicWall>>(pd, pg2, sys,
							     make_shared<HarmonicWall>(-box.boxSize.z*0.25));
  auto extForces2 = make_shared<ExternalForces<HarmonicWall>>(pd, pg3, sys,
							      make_shared<HarmonicWall>(box.boxSize.z*0.25));
  // auto extForces3 = make_shared<ExternalForces<HarmonicWall>>(pd, pg4, sys, HarmonicWall(-20));

  //Add interactors to integrator.
  verlet->addInteractor(extForces);
  verlet->addInteractor(extForces2);
  // verlet->addInteractor(extForces3);

  //Modules working with pairs of particles usually ask for a Potential object
  //PairForces decides if it should use a neighbour list or treat the system as NBody,
  //You can force the use of a certain neighbour list passing its name as a second template argument

  using PairForces = PairForces<Potential::LJ, VerletList>;

  //This is the general interface for setting up a potential
  auto pot = make_shared<Potential::LJ>(sys);
  {
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
  }

  PairForces::Parameters params;
  params.box = box;  //Box to work on
  auto pairforces = make_shared<PairForces>(pd, pg, sys, params, pot);


  //This NBody module can be used to check the correctness of the PairForces short range results.
  //Forces the interaction to be processed as an Nbody O(N^2) computation.
  // using NBodyForces = NBodyForces<Potential::LJ>;
  // NBodyForces::Parameters nbodyPar;
  // nbodyPar.box = box;
  // auto nbody = make_shared<NBodyForces>(pd,pg,sys, nbodyPar ,pot);

  //You can add as many modules as necessary
  verlet->addInteractor(pairforces);
    //verlet->addInteractor(nbody);


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
  std::ofstream out(outputFile);
  Timer tim;
  tim.tic();
  //Run the simulation
  forj(0, numberSteps){
    //This will instruct the integrator to take the simulation to the next time step,
    //whatever that may mean for the particular integrator (i.e compute forces and update positions once)
    verlet->forwardTime();

    //Write results
    if(printSteps>0 and j%printSteps==0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      //continue;
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<endl;
      real3 p;
      fori(0, numberParticles){
	real4 pc = pos[sortedIndex[i]];
	p = box.apply_pbc(make_real3(pc));
	int type = pc.w;
	out<<p<<" "<<0.5*(type==1?2:1)*pow(2,1/6.)<<" "<<type<<"\n";
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
      real visco = 1/(6*M_PI);
      std::ofstream default_options(file);
      default_options<<"boxSize 90 90 90"<<std::endl;
      default_options<<"numberParticles 16384"<<std::endl;
      default_options<<"dt 0.01"<<std::endl;
      default_options<<"numberSteps 20000"<<std::endl;
      default_options<<"printSteps 100"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
      default_options<<"temperature 0.1"<<std::endl;
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
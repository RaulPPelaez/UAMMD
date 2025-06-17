/*Raul P. Pelaez 2019-2021. A Brownian Dynamics with Hydrodynamic Interactions
simulation.

Runs a Brownian Hydrodynamics simulation with particles starting in a box,
particles have different radius and interact with a LJ potential.

Reads some parameters from a file called "data.main.bdhi", if not present it
will be auto generated with some default parameters.

*/

// This include contains the basic needs for an uammd project
#include "uammd.cuh"
// The rest can be included depending on the used modules
#include "Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include "Integrator/BDHI/BDHI_Lanczos.cuh"
#include "Interactor/NeighbourList/CellList.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/Potential/Potential.cuh"
#include "utils/InitialConditions.cuh"
#include "utils/InputFile.h"
#include <fstream>

using namespace uammd;
using std::endl;
using std::make_shared;
real3 boxSize;
real dt;
std::string outputFile;
int numberParticles;
int numberSteps, printSteps;

real temperature, viscosity;

void readParameters(std::string file);

int main(int argc, char *argv[]) {
  // UAMMD System entity holds information about the GPU and tools to interact
  // with the computer itself (such as a loging system). All modules need a
  // System to work on.

  auto sys = make_shared<System>(argc, argv);
  readParameters("data.main.bdhi");
  // Modules will ask System when they need a random number (i.e for seeding the
  // GPU RNG).
  ullint seed = 0xf31337Bada55D00dULL ^ time(NULL);
  sys->rng().setSeed(seed);
  // ParticleData stores all the needed properties the simulation will need.
  // Needs to start with a certain number of particles, which can be changed
  // mid-simulation
  auto pd = make_shared<ParticleData>(numberParticles, sys);
  // Some modules need a simulation box (i.e PairForces for the PBC)
  Box box(boxSize);
  // Initial positions
  {
    // Ask pd for a property like so:
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    auto radius = pd->getRadius(access::location::cpu, access::mode::write);
    // Start in a fcc lattice, pos.w contains the particle type
    auto initial = initLattice(box.boxSize * 0.5, numberParticles, fcc);
    // Copy initial positions to pos, pos.w contains type, set it to 0 or 1
    // randomly
    std::transform(initial.begin(), initial.end(), pos.begin(), [&](real4 p) {
      p.w = sys->rng().uniform(0, 1) > 0.5 ? 0 : 1;
      return p;
    });
    // Set particles with type 0 to have radius 1 and type 1 to radius 0.5.
    std::transform(pos.begin(), pos.end(), radius.begin(),
                   [&](real4 p) { return p.w == 1 ? 1 : 0.5; });
  }

  // Initialize Integrator module
  BDHI::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.dt = dt;
  par.tolerance = 1e-3;
  auto bdhi = make_shared<BDHI::EulerMaruyama<BDHI::Lanczos>>(pd, par);
  // Initialize Interactor module
  {
    using PairForces = PairForces<Potential::LJ>;
    // This is the general interface for setting up a potential
    auto pot = make_shared<Potential::LJ>();
    {
      // Each Potential describes the pair interactions with certain parameters.
      // The needed ones are in InputPairParameters inside each potential, in
      // this case:
      Potential::LJ::InputPairParameters par;
      par.epsilon = 1.0;
      par.shift = false;

      par.sigma = 2;
      par.cutOff = 2.5 * par.sigma;
      // Once the InputPairParameters has been filled accordingly for a given
      // pair of types, a potential can be informed like this:
      pot->setPotParameters(1, 1, par);

      par.sigma = 1.0;
      par.cutOff = 2.5 * par.sigma;
      pot->setPotParameters(0, 0, par);

      par.sigma = 0.5 * (2.0 + 1.0);
      par.cutOff = 2.5 * par.sigma;
      // the pair 1,0 is registered as well with this call, and assumed to be
      // the same
      pot->setPotParameters(0, 1, par);
    }

    PairForces::Parameters params;
    params.box = box; // Box to work on
    auto pairforces = make_shared<PairForces>(pd, params, pot);
    bdhi->addInteractor(pairforces);
  }
  // You can issue a logging event like this, a wide variety of log levels
  // exists (see System.cuh). A maximum log level is set in System.cuh, every
  // logging event with a level superior to the max will result in
  //  absolutely no overhead, so dont be afraid to write System::DEBUGX log
  //  calls.
  sys->log<System::MESSAGE>("RUNNING!!!");
  // Ask ParticleData to sort the particles in memory
  // It is a good idea to sort the particles once in a while during the
  // simulation This can increase performance considerably as it improves
  // coalescence. Sorting the particles will cause the particle arrays to change
  // in order and (possibly) address. This changes will be informed with signals
  // and any module that needs to be aware of such changes will acknowedge it
  // through a callback (see ParticleData.cuh).
  pd->sortParticles();

  std::ofstream out(outputFile);
  Timer tim;
  tim.tic();

  // Run the simulation
  forj(0, numberSteps) {
    // This will instruct the integrator to take the simulation to the next time
    // step, whatever that may mean for the particular integrator (i.e compute
    // forces and update positions once)
    bdhi->forwardTime();

    // Write results
    if (printSteps and j % printSteps == 0) {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      auto radius = pd->getRadius(access::location::cpu, access::mode::read);
      // This allows to access the particles with the starting order so the
      // particles are written in the same order
      //  even after a sorting
      const int *sortedIndex = pd->getIdOrderedIndices(access::location::cpu);

      out << "#" << endl;
      real3 p;
      fori(0, numberParticles) {
        int si = sortedIndex[i];
        real4 pc = pos[si];
        p = make_real3(pc);
        int type = int(radius[si]);
        out << p << " " << radius[si] << " " << type << "\n";
      }
    }
    // Sort the particles every few steps
    // It is not an expensive thing to do really.
    if (j % 500 == 0) {
      pd->sortParticles();
    }
  }

  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", numberSteps / totalTime);
  // sys->finish() will ensure a smooth termination of any UAMMD module.
  sys->finish();

  return 0;
}

void readParameters(std::string file) {

  {
    if (!std::ifstream(file).good()) {
      std::ofstream default_options(file);
      real vis = 1 / (6 * M_PI);
      default_options << "boxSize 128 128 128" << std::endl;
      default_options << "numberParticles 8000" << std::endl;
      default_options << "dt 0.001" << std::endl;
      default_options << "numberSteps 100000" << std::endl;
      default_options << "printSteps 100" << std::endl;
      default_options << "outputFile /dev/stdout" << std::endl;
      default_options << "temperature 1.0" << std::endl;
      default_options << "viscosity " << vis << std::endl;
    }
  }

  InputFile in(file);

  in.getOption("boxSize", InputFile::Required) >> boxSize.x >> boxSize.y >>
      boxSize.z;
  in.getOption("numberSteps", InputFile::Required) >> numberSteps;
  in.getOption("printSteps", InputFile::Required) >> printSteps;
  in.getOption("dt", InputFile::Required) >> dt;
  in.getOption("numberParticles", InputFile::Required) >> numberParticles;
  in.getOption("outputFile", InputFile::Required) >> outputFile;

  in.getOption("temperature", InputFile::Required) >> temperature;
  in.getOption("viscosity", InputFile::Required) >> viscosity;
}

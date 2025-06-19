/*Raul P. Pelaez 2019-2021. An SPH example.

Particles start in a cube configuration and are inside a prism box (longer in
z). There is gravity and the bottom wall of the box is repulsive. The SPH
particles will fall until hitting the bottom and then flow around until the
energy from the fall is dissipated.

Needs a data.main.sph parameter file, if not present it will be auto generated
with some default parameters.

You can visualize the reuslts with superpunto


 */

// This include contains the basic needs for an uammd project
#include "uammd.cuh"
// The rest can be included depending on the used modules
#include "Integrator/VerletNVE.cuh"
#include "Interactor/ExternalForces.cuh"
#include "Interactor/SPH.cuh"
#include "utils/ForceEnergyVirial.cuh"
#include "utils/InitialConditions.cuh"
#include "utils/InputFile.h"
#include <fstream>

using namespace uammd;

using std::endl;
using std::make_shared;
// An Wall+Gravity functor to be used in a ExternalForces module (See
// ExternalForces.cuh)
struct Wall {
  real k = 20;
  real3 L;
  Wall(real3 L) : L(L) {}

  __device__ ForceEnergyVirial sum(Interactor::Computables comp,
                                   const real4 &pos) {
    real3 f = real3();
    if (pos.x < -L.x * 0.5f)
      f.x = k;
    if (pos.x > L.x * 0.5f)
      f.x = -k;
    if (pos.y < -L.y * 0.5f)
      f.y = k;
    if (pos.y > L.y * 0.5f)
      f.y = -k;
    if (pos.z < -L.z * 0.5f)
      f.z = k;
    if (pos.z > L.z * 0.5f)
      f.z = -k;

    f.z += k / 30.0f;
    ForceEnergyVirial fev{};
    fev.force = f;
    return fev;
  }

  auto getArrays(ParticleData *pd) {
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    return std::make_tuple(pos.raw());
  }
};

int N;
real3 L;
std::string outputFile;
real dt;
int nsteps, printSteps;
real viscosity, gasStiffness, restDensity, support;

void readParameters(std::string file);

int main(int argc, char *argv[]) {
  std::cerr << "ERROR, I need some parameters!!\nTry to run me with:\n./"
            << argv[0] << " 14 45 0.01 0.9 20000 500 -20" << std::endl;
  // UAMMD System entity holds information about the GPU and tools to interact
  // with the computer itself (such as a loging system). All modules need a
  // System to work on.
  auto sys = make_shared<System>(argc, argv);
  readParameters("data.main.sph");
  // Modules will ask System when they need a random number (i.e for seeding the
  // GPU RNG).
  ullint seed = 0xf31337Bada55D00dULL ^ time(NULL);
  sys->rng().setSeed(seed);
  // ParticleData stores all the needed properties the simulation will need.
  // Needs to start with a certain number of particles, which can be changed
  // mid-simulation
  auto pd = make_shared<ParticleData>(N, sys);
  // Some modules need a simulation box (i.e PairForces for the PBC)
  Box box(L);
  // Initial positions
  {
    // Ask pd for a property like so, do not store this handle, let it out of
    // scope ASAP:
    auto pos = pd->getPos(access::cpu, access::write);
    // returns an array with positions in an FCC lattice
    auto initial = initLattice(make_real3(box.boxSize.x * 0.9), N, fcc);
    // Copy to pos
    std::transform(initial.begin(), initial.end(), pos.begin(), [&](real4 p) {
      p += make_real4(0, 0, -20, 0);
      p.w = 0;
      return p;
    });
  }
  // Modules can work on a certain subset of particles if needed, the particles
  // can be grouped following any criteria The builtin ones will generally work
  // faster than a custom one. See ParticleGroup.cuh for a list A group created
  // with no criteria will contain all the particles
  auto pg = make_shared<ParticleGroup>(pd, "All");
  std::ofstream out(outputFile);
  // Some modules need additional parameters, in this case VerletNVT needs dt,
  // temperature... When additional parameters are needed, they need to be
  // supplied in a form similar to this:
  {
    auto vel = pd->getVel(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par, vel.begin(), vel.end(), real3());
  }
  VerletNVE::Parameters par;
  par.dt = dt;
  // If set to true (default), VerletNVE will compute Energy at step 0 and
  // modify the velocities accordingly
  par.initVelocities = false;
  auto verlet = make_shared<VerletNVE>(pg, par);
  {
    // Harmonic walls acting on different particle groups
    // This two interactors will cause particles in group pg2 to stick to a wall
    // in -Lz/4 And the ones in pg3 to +Lz/4
    auto extForces =
        make_shared<ExternalForces<Wall>>(pg, make_shared<Wall>(box.boxSize));
    // Add interactors to integrator.
    verlet->addInteractor(extForces);
  }
  SPH::Parameters params;
  params.box = box; // Box to work on
  // These are the default parameters,
  // if any parameter is not present, it will revert to the default in the .cuh
  params.support = support;
  params.viscosity = viscosity;
  params.gasStiffness = gasStiffness;
  params.restDensity = restDensity;
  auto sph = make_shared<SPH>(pg, params);
  // You can add as many modules as necessary
  verlet->addInteractor(sph);
  // You can issue a logging event like this, a wide variety of log levels
  // exists (see System.cuh). A maximum log level is set in System.cuh, every
  // logging event with a level superior to the max will result in
  //  absolutely no overhead, so dont be afraid to write System::DEBUGX log
  //  calls.
  sys->log<System::MESSAGE>("RUNNING!!!");
  // Ask ParticleData to sort the particles in memory!
  // It is a good idea to sort the particles once in a while during the
  // simulation This can increase performance considerably as it improves
  // coalescence. Sorting the particles will cause the particle arrays to change
  // in order and (possibly) address. This changes will be informed with signals
  // and any module that needs to be aware of such changes will acknowledge it
  // through a callback (see ParticleData.cuh).
  pd->sortParticles();
  Timer tim;
  tim.tic();
  // Run the simulation
  forj(0, nsteps) {
    // This will instruct the integrator to take the simulation to the next time
    // step, whatever that may mean for the particular integrator (i.e compute
    // forces and update positions once)
    verlet->forwardTime();
    // Write results
    if (printSteps and j % printSteps == 0) {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      // continue;
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int *sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out << "#Lx=" << 0.5 * box.boxSize.x << ";Ly=" << 0.5 * box.boxSize.y
          << ";Lz=" << 0.5 * box.boxSize.z << ";" << endl;
      real3 p;
      fori(0, N) {
        real4 pc = pos.raw()[sortedIndex[i]];
        p = box.apply_pbc(make_real3(pc));
        int type = pc.w;
        out << p << " " << 0.5 * (type == 1 ? 2 : 1) << " " << type << "\n";
      }
    }
    // Sort the particles every few steps
    // It is not an expensive thing to do really.
    if (j % 500 == 0) {
      pd->sortParticles();
    }
  }
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps / totalTime);
  // sys->finish() will ensure a smooth termination of any UAMMD module.
  sys->finish();
  return 0;
}

void readParameters(std::string file) {

  {
    if (!std::ifstream(file).good()) {
      std::ofstream default_options(file);
      default_options << "boxSize 45 45 90" << std::endl;
      default_options << "numberParticles 16384" << std::endl;
      default_options << "dt 0.01" << std::endl;
      default_options << "numberSteps 100000" << std::endl;
      default_options << "printSteps 100" << std::endl;
      default_options << "outputFile /dev/stdout" << std::endl;
      default_options << "viscosity 10" << std::endl;
      default_options << "gasStiffness 60" << std::endl;
      default_options << "support 2.4" << std::endl; // 1
      default_options << "restDensity 0.3" << std::endl;
    }
  }

  InputFile in(file);

  in.getOption("boxSize", InputFile::Required) >> L.x >> L.y >> L.z;
  in.getOption("numberSteps", InputFile::Required) >> nsteps;
  in.getOption("printSteps", InputFile::Required) >> printSteps;
  in.getOption("dt", InputFile::Required) >> dt;
  in.getOption("numberParticles", InputFile::Required) >> N;
  in.getOption("outputFile", InputFile::Required) >> outputFile;
  in.getOption("viscosity", InputFile::Required) >> viscosity;
  in.getOption("gasStiffness", InputFile::Required) >> gasStiffness;
  in.getOption("support", InputFile::Required) >> support;
  in.getOption("restDensity", InputFile::Required) >> restDensity;
}

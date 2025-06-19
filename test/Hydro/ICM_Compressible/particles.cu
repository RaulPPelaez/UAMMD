/* Raul P. Pelaez 2022. Compressible ICM particle-related test code.
   Can use an ICM or BD Integrator and include LJ interactions.
   If LJ interactions (sterics) are enabled particles start in an FCC lattice,
   otherwise particles are distributed randomly. Reads parameters from a
   data.main.
 */
#include "Integrator/BrownianDynamics.cuh"
#include "Integrator/Hydro/ICM_Compressible.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/Potential/Potential.cuh"
#include "computeStructureFactor.cuh"
#include "uammd.cuh"
#include "utils/InitialConditions.cuh"
#include "utils/InputFile.h"
#include <cstdint>
#include <memory>
#include <stdexcept>

using namespace uammd;

using ICM = Hydro::ICM_Compressible;

struct Parameters {
  std::string scheme;
  real dt = 0.1;
  real3 boxSize = make_real3(32, 32, 32) * 100;
  int3 cellDim = {30, 30, 30};
  real bulkViscosity = 127.05;
  real speedOfSound = 14.67;
  real shearViscosity = 53.71;
  real temperature = 1;
  real initialDensity = 0.632;

  real relaxTime = 500;
  real simulationTime = -1;
  real printTime = 0;

  int numberParticles = 16384;
  bool enableSterics = false;
  real epsilon = 1;
};

auto createICMIntegratorCompressible(std::shared_ptr<ParticleData> pd,
                                     Parameters ipar) {
  ICM::Parameters par;
  par.dt = ipar.dt;
  par.boxSize = ipar.boxSize;
  // par.hydrodynamicRadius = 0.5;
  par.cellDim = ipar.cellDim;
  par.bulkViscosity = ipar.bulkViscosity;
  par.speedOfSound = ipar.speedOfSound;
  par.shearViscosity = ipar.shearViscosity;
  par.temperature = ipar.temperature;
  par.initialDensity = [=](real3 r) { return ipar.initialDensity; };
  // par.initialVelocityX = [=](real3 r){return
  // 0.001*ipar.boxSize.x/(ipar.cellDim.x);};
  return std::make_shared<ICM>(pd, par);
}

auto createBDIntegrator(std::shared_ptr<ParticleData> pd, Parameters ipar) {
  BD::EulerMaruyama::Parameters par;
  par.dt = ipar.dt;
  par.hydrodynamicRadius = 0.91 * ipar.boxSize.x / ipar.cellDim.x;
  par.viscosity = ipar.shearViscosity;
  par.temperature = ipar.temperature;
  return std::make_shared<BD::EulerMaruyama>(pd, par);
}

auto createIntegrator(std::shared_ptr<ParticleData> pd, Parameters ipar) {
  std::shared_ptr<Integrator> inte;
  if (ipar.scheme == "bd") {
    inte = createBDIntegrator(pd, ipar);
  } else if (ipar.scheme == "icm_compressible") {
    inte = createICMIntegratorCompressible(pd, ipar);
  } else {
    System::log<System::EXCEPTION>("Invalid scheme");
    throw std::runtime_error("Invalid scheme");
  }
  return inte;
}

auto createLJInteractor(std::shared_ptr<ParticleData> pd, Parameters ipar) {
  using PairForces = PairForces<Potential::LJ>;
  auto pot = std::make_shared<Potential::LJ>();
  {
    Potential::LJ::InputPairParameters par;
    par.epsilon = ipar.epsilon;
    par.shift = false;
    par.sigma =
        ipar.boxSize.x / ipar.cellDim.x; // Approx twice the hydrodynamicRadius
    par.cutOff = 2.5 * par.sigma;
    pot->setPotParameters(0, 0, par);
  }
  PairForces::Parameters params;
  params.box = Box(ipar.boxSize); // Box to work on
  auto pairforces = std::make_shared<PairForces>(pd, params, pot);
  return pairforces;
}

Parameters readParameters(std::string file) {
  InputFile in(file);
  Parameters par;
  in.getOption("dt", InputFile::Required) >> par.dt;
  in.getOption("boxSize", InputFile::Required) >> par.boxSize.x >>
      par.boxSize.y >> par.boxSize.z;
  in.getOption("cellDim", InputFile::Required) >> par.cellDim.x >>
      par.cellDim.y >> par.cellDim.z;
  in.getOption("bulkViscosity", InputFile::Required) >> par.bulkViscosity;
  in.getOption("shearViscosity", InputFile::Required) >> par.shearViscosity;
  in.getOption("speedOfSound", InputFile::Required) >> par.speedOfSound;
  in.getOption("temperature", InputFile::Required) >> par.temperature;
  in.getOption("initialDensity", InputFile::Required) >> par.initialDensity;
  in.getOption("relaxTime", InputFile::Optional) >> par.relaxTime;
  in.getOption("simulationTime", InputFile::Required) >> par.simulationTime;
  in.getOption("printTime", InputFile::Required) >> par.printTime;
  in.getOption("numberParticles", InputFile::Required) >> par.numberParticles;
  in.getOption("scheme", InputFile::Required) >> par.scheme;
  par.enableSterics = bool(in.getOption("enableSterics", InputFile::Optional));
  if (par.enableSterics) {
    in.getOption("epsilon", InputFile::Required) >> par.epsilon;
  }
  return par;
}

void writePositions(std::shared_ptr<ParticleData> pd) {
  auto pos = pd->getPos(access::cpu, access::read);
  std::cout << "#" << std::endl;
  for (auto p : pos)
    std::cout << make_real3(p) << "\n";
  std::cout << std::flush;
}

class Random {
  uint s1, s2;
  real L;

public:
  Random(uint s1, uint s2, real L) : s1(s1), s2(s2), L(L) {}

  __device__ real4 operator()(int i) {
    Saru rng(s1, s2, i);
    return (make_real4(rng.f(), rng.f(), rng.f(), 0) - 0.5) * L;
  }
};

auto initializeParticles(Parameters par, std::shared_ptr<System> sys) {
  int numberParticles = par.numberParticles;
  auto pd = std::make_shared<ParticleData>(numberParticles, sys);
  auto pos = pd->getPos(access::gpu, access::write);
  if (par.enableSterics) {
    auto init = initLattice(par.boxSize, numberParticles, fcc);
    thrust::copy(init.begin(), init.end(),
                 thrust::device_ptr<real4>(pos.begin()));
  } else {
    auto cit = thrust::make_counting_iterator(0);
    thrust::transform(
        thrust::cuda::par, cit, cit + numberParticles, pos.begin(),
        Random(sys->rng().next32(), sys->rng().next32(), par.boxSize.x));
    auto constant = thrust::make_constant_iterator(make_real4(0, 0, 0, 0));
    thrust::copy(thrust::cuda::par, constant, constant + 1, pos.begin());
  }
  return pd;
}

int main(int argc, char *argv[]) {
  auto sys = std::make_shared<System>(argc, argv);
  auto par = readParameters(argv[1]);
  auto pd = initializeParticles(par, sys);
  auto integrator = createIntegrator(pd, par);
  if (par.enableSterics)
    integrator->addInteractor(createLJInteractor(pd, par));
  {
    int relaxSteps = par.relaxTime / par.dt + 1;
    fori(0, relaxSteps) integrator->forwardTime();
  }
  int ntimes = par.simulationTime / par.dt;
  int sampleSteps = par.printTime / par.dt;
  Timer tim;
  tim.tic();
  fori(0, ntimes) {
    integrator->forwardTime();
    if (i % sampleSteps == 0) {
      writePositions(pd);
    }
  }
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", ntimes / totalTime);
  return 0;
}

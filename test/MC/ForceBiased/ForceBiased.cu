/*Raul P. Pelaez 2020. Force biased Monte Carlo test code.

  Runs a simulation with particles interacting through a short range potential
(either Lennard-Jones or a soft gaussian like potential). The simulation can be
forwarded using Brownian Dynamics, Langevin dynamics or a Force Biased Monte
Carlo algorithm.

Particles start in a simple cubic lattice. The code outputs the positions of all
particles and the total system's internal and kinectic energy.

  Requires a data.main with the parameters (see the readParameters function).

  Intended to be used by ForceBiased/test.bash
 */
#include "Integrator/BrownianDynamics.cuh"
#include "Integrator/MonteCarlo/ForceBiased.cuh"
#include "Integrator/VerletNVT.cuh"
#include "Interactor/NeighbourList/CellList.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/Potential/Potential.cuh"
#include "uammd.cuh"
#include "utils/InitialConditions.cuh"
#include "utils/InputFile.h"

using namespace uammd;

struct SoftPotentialFunctor {
  struct InputPairParameters {
    real cutOff;
    real b, d, U0;
  };

  struct __align__(16) PairParameters {
    real cutOff2;
    real b, d, U0;
  };

  __device__ real force(real r2, PairParameters params) {
    if (r2 >= params.cutOff2 or r2 == 0)
      return 0;
    const real r = sqrtf(r2);
    const real d = params.d;
    const real b = params.b;
    const real U0 = params.U0;
    if (r < d) {
      return -U0 / (b * r);
    } else {
      return -U0 * exp((d - r) / b) / (b * r);
    }
  }

  __device__ real energy(real r2, PairParameters params) {
    if (r2 >= params.cutOff2 or r2 == 0)
      return 0;
    const real r = sqrtf(r2);
    const real d = params.d;
    const real b = params.b;
    const real U0 = params.U0;
    const real rdb = (d - r) / b;
    if (r < d) {
      return (U0 + U0 * rdb);
    } else {
      return U0 * exp(rdb);
    }
  }

  static PairParameters processPairParameters(InputPairParameters in_par) {
    PairParameters params;
    params.cutOff2 = in_par.cutOff * in_par.cutOff;
    params.b = in_par.b;
    params.d = in_par.d;
    params.U0 = in_par.U0;
    return params;
  }
};

using SoftPotential = Potential::Radial<SoftPotentialFunctor>;

struct Parameters {
  real3 boxSize;
  int numberSteps;
  int printSteps;
  int relaxSteps;
  real dt;
  int numberParticles;
  real cutOff;
  real temperature;
  std::string outfile, energyOutfile;

  std::string potential, integrator;
  // LJ
  real sigma;
  real epsilon;
  bool shift;
  // Soft
  real U0, d, b;

  // Force biased
  real acceptanceRatio;
};

struct UAMMD {
  std::shared_ptr<ParticleData> pd;
  Parameters parameters;
};

UAMMD initialize(int argc, char *argv[]);

std::shared_ptr<Integrator> createIntegrator(UAMMD sim);

std::shared_ptr<Interactor> createShortRangeInteractor(UAMMD sim);

template <class NVT> void runSimulation(UAMMD sim, std::shared_ptr<NVT> mc);

int main(int argc, char *argv[]) {
  auto sim = initialize(argc, argv);
  std::shared_ptr<Integrator> mc;
  mc = createIntegrator(sim);
  mc->addInteractor(createShortRangeInteractor(sim));
  Timer tim;
  tim.tic();
  runSimulation(sim, mc);
  auto totalTime = tim.toc();
  System::log<System::MESSAGE>("mean FPS: %.2f",
                               sim.parameters.numberSteps / totalTime);
  return 0;
}

Parameters readParameters(std::string datamain);

void initializePositions(UAMMD sim) {
  auto pos = sim.pd->getPos(access::location::cpu, access::mode::write);
  auto initial =
      initLattice(sim.parameters.boxSize, sim.parameters.numberParticles, sc);
  std::copy(initial.begin(), initial.end(), pos.begin());
}

UAMMD initialize(int argc, char *argv[]) {
  UAMMD sim;
  auto sys = std::make_shared<System>(argc, argv);
  ullint seed = 0xf31337Bada55D00dULL ^ time(NULL);
  sys->rng().setSeed(seed);
  sim.parameters = readParameters("data.main");
  sim.pd = std::make_shared<ParticleData>(sim.parameters.numberParticles, sys);
  initializePositions(sim);
  return sim;
}

std::shared_ptr<MC::ForceBiased> createIntegratorForceBiased(UAMMD sim) {
  typename MC::ForceBiased::Parameters par;
  par.beta = 1.0 / sim.parameters.temperature;
  par.stepSize = sim.parameters.dt;
  par.acceptanceRatio = sim.parameters.acceptanceRatio;
  return std::make_shared<MC::ForceBiased>(sim.pd, par);
}

std::shared_ptr<BD::EulerMaruyama> createIntegratorBD(UAMMD sim) {
  typename BD::EulerMaruyama::Parameters par;
  par.temperature = sim.parameters.temperature;
  par.viscosity = 1 / (6 * M_PI * 2);
  par.hydrodynamicRadius = 0.5;
  par.dt = sim.parameters.dt;
  return std::make_shared<BD::EulerMaruyama>(sim.pd, par);
}

std::shared_ptr<VerletNVT::GronbechJensen> createIntegratorMD(UAMMD sim) {
  typename VerletNVT::GronbechJensen::Parameters par;
  par.temperature = sim.parameters.temperature;
  par.friction = 1 / (6 * M_PI * 2);
  par.dt = sim.parameters.dt;
  return std::make_shared<VerletNVT::GronbechJensen>(sim.pd, par);
}

std::shared_ptr<Integrator> createIntegrator(UAMMD sim) {
  if (sim.parameters.integrator.compare("bd") == 0) {
    return createIntegratorBD(sim);
  } else if (sim.parameters.integrator.compare("forcebiased") == 0) {
    return createIntegratorForceBiased(sim);
  } else if (sim.parameters.integrator.compare("md") == 0) {
    return createIntegratorMD(sim);
  } else {
    System::log<System::CRITICAL>(
        "Unrecognized Integrator. Use forcebiased, bd or md");
    return nullptr;
  }
}

template <class UsePotential>
std::shared_ptr<UsePotential> createPotential(UAMMD sim);

template <> std::shared_ptr<Potential::LJ> createPotential(UAMMD sim) {
  auto pot = std::make_shared<Potential::LJ>();
  Potential::LJ::InputPairParameters ppar;
  ppar.epsilon = sim.parameters.epsilon;
  ppar.shift = sim.parameters.shift;
  ppar.sigma = sim.parameters.sigma;
  ppar.cutOff = sim.parameters.cutOff * sim.parameters.sigma;
  pot->setPotParameters(0, 0, ppar);
  return pot;
}

template <> std::shared_ptr<SoftPotential> createPotential(UAMMD sim) {
  auto pot = std::make_shared<SoftPotential>();
  SoftPotential::InputPairParameters ppar;
  ppar.cutOff = sim.parameters.cutOff;
  ppar.U0 = sim.parameters.U0;
  ppar.b = sim.parameters.b;
  ppar.d = sim.parameters.d;
  pot->setPotParameters(0, 0, ppar);
  return pot;
}

template <class UsePotential>
std::shared_ptr<Interactor> createShortRangeInteractor_impl(UAMMD sim) {
  auto pot = createPotential<UsePotential>(sim);
  using SR = PairForces<UsePotential>;
  typename SR::Parameters params;
  params.box = Box(sim.parameters.boxSize);
  auto pairForces = std::make_shared<SR>(sim.pd, params, pot);
  return pairForces;
}

std::shared_ptr<Interactor> createShortRangeInteractor(UAMMD sim) {
  if (sim.parameters.potential.compare("LJ") == 0) {
    return createShortRangeInteractor_impl<Potential::LJ>(sim);
  } else if (sim.parameters.potential.compare("soft") == 0) {
    return createShortRangeInteractor_impl<SoftPotential>(sim);
  } else {
    System::log<System::CRITICAL>("Unrecognized potential. Use LJ or soft");
    return nullptr;
  }
}

template <class NVT> real getCurrentEnergy(UAMMD sim, std::shared_ptr<NVT> mc) {
  {
    auto energy = sim.pd->getEnergy(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par, energy.begin(), energy.end(), real());
  }
  auto interactors = mc->getInteractors();
  for (auto &i : interactors) {
    i->sum({false, true, false});
  }
  auto energy = sim.pd->getEnergy(access::location::gpu, access::mode::read);
  auto currentEnergy =
      thrust::reduce(thrust::cuda::par, energy.begin(), energy.end());
  return 0.5 * currentEnergy / sim.pd->getNumParticles();
}

template <class NVT> class SimulationWriter {
  std::ofstream out;
  std::ofstream eout;
  UAMMD sim;
  std::shared_ptr<NVT> mc;

public:
  SimulationWriter(UAMMD sim, std::shared_ptr<NVT> mc) : sim(sim), mc(mc) {
    out.open(sim.parameters.outfile);
    eout.open(sim.parameters.energyOutfile);
    eout.precision(2 * sizeof(uammd::real));
    out.precision(2 * sizeof(uammd::real));
  }

  void writeCurrent() {
    System::log<System::DEBUG1>("[System] Writing to disk...");
    auto par = sim.parameters;
    auto pos = sim.pd->getPos(access::location::cpu, access::mode::read);
    const int *sortedIndex = sim.pd->getIdOrderedIndices(access::location::cpu);
    out << "#Lx=" << 0.5 * par.boxSize.x << ";Ly=" << 0.5 * par.boxSize.y
        << ";Lz=" << 0.5 * par.boxSize.z << ";" << std::endl;
    real3 p;
    fori(0, par.numberParticles) {
      real4 pc = pos[sortedIndex[i]];
      p = make_real3(pc);
      out << p << " 0.5 0\n";
    }
    double U = getCurrentEnergy(sim, mc);
    double K = 1.5 * par.temperature;
    eout << U << " " << (K) << std::endl;
    out << std::flush;
  }
};

template <class NVT> void runSimulation(UAMMD sim, std::shared_ptr<NVT> mc) {
  SimulationWriter<NVT> write(sim, mc);
  auto par = sim.parameters;
  forj(0, par.relaxSteps) { mc->forwardTime(); }
  forj(0, par.numberSteps) {
    mc->forwardTime();
    if (par.printSteps > 0 and j % par.printSteps == 0) {
      write.writeCurrent();
    }
  }
}

Parameters readParameters(std::string datamain) {
  InputFile in(datamain);
  Parameters par;
  in.getOption("potential", InputFile::Required) >> par.potential;
  in.getOption("integrator", InputFile::Required) >> par.integrator;
  in.getOption("boxSize", InputFile::Required) >> par.boxSize.x >>
      par.boxSize.y >> par.boxSize.z;
  in.getOption("numberSteps", InputFile::Required) >> par.numberSteps;
  in.getOption("printSteps", InputFile::Required) >> par.printSteps;
  in.getOption("relaxSteps", InputFile::Required) >> par.relaxSteps;
  in.getOption("dt", InputFile::Required) >> par.dt;
  in.getOption("numberParticles", InputFile::Required) >> par.numberParticles;
  in.getOption("temperature", InputFile::Required) >> par.temperature;
  in.getOption("outfile", InputFile::Required) >> par.outfile;
  in.getOption("energyOutfile", InputFile::Required) >> par.energyOutfile;
  in.getOption("cutOff", InputFile::Required) >> par.cutOff;
  if (par.potential.compare("LJ") == 0) {
    in.getOption("sigma", InputFile::Required) >> par.sigma;
    in.getOption("epsilon", InputFile::Required) >> par.epsilon;
    in.getOption("shiftLJ", InputFile::Required) >> par.shift;
  } else if (par.potential.compare("soft") == 0) {
    in.getOption("U0", InputFile::Required) >> par.U0;
    in.getOption("b", InputFile::Required) >> par.b;
    in.getOption("d", InputFile::Required) >> par.d;
  }
  if (par.integrator.compare("forcebiased") == 0) {
    in.getOption("acceptanceRatio", InputFile::Required) >> par.acceptanceRatio;
  }

  return par;
}

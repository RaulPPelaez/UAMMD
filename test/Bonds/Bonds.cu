
#include "Integrator/BrownianDynamics.cuh"
#include "Interactor/BondedForces.cuh"
#include "utils/InputFile.h"
#include <uammd.cuh>

using namespace uammd;

struct Parameters {
  std::string initpos;
  std::string bondfile;
  real dt;
  real temperature;
  real simulationTime;
  real printTime;
  real3 lbox;
};

Parameters readParameters(std::string datamain);

auto initializeParticles(std::shared_ptr<System> sys, Parameters par) {
  std::ifstream in(par.initpos);
  std::istream_iterator<real3> begin(in), end;
  std::vector<real3> posFromFile(begin, end);
  int numberParticles = posFromFile.size();
  auto pd = std::make_shared<ParticleData>(numberParticles, sys);
  auto pos = pd->getPos(access::cpu, access::write);
  std::transform(posFromFile.begin(), posFromFile.end(), pos.begin(),
                 [](real3 p) { return make_real4(p); });
  return pd;
}

auto createBD(std::shared_ptr<ParticleData> pd, Parameters par) {
  using BD = BD::EulerMaruyama;
  BD::Parameters bdpar;
  bdpar.dt = par.dt;
  bdpar.hydrodynamicRadius = 1;
  bdpar.viscosity = 1 / (6 * M_PI);
  bdpar.temperature = par.temperature;
  return std::make_shared<BD>(pd, bdpar);
}

auto createBonds(std::shared_ptr<ParticleData> pd, Parameters par) {
  using Bond = BondedType::Harmonic;
  using BF = BondedForces<Bond, 2>;
  BF::Parameters bpar;
  bpar.file = par.bondfile;
  return std::make_shared<BF>(pd, bpar);
}

void writeSimulation(std::shared_ptr<ParticleData> pd) {
  auto &out = std::cout;
  auto pos = pd->getPos(access::cpu, access::read);
  out << "#" << "\n";
  for (auto p : pos) {
    out << p << "\n";
  }
  out << std::flush;
}

int main(int argc, char *argv[]) {
  auto sys = std::make_shared<System>(argc, argv);
  auto par = readParameters("data.main");
  auto pd = initializeParticles(sys, par);
  auto bd = createBD(pd, par);
  bd->addInteractor(createBonds(pd, par));
  int nsteps = par.simulationTime / par.dt;
  int printSteps = par.printTime / par.dt;
  fori(0, nsteps) {
    bd->forwardTime();
    if (i % printSteps == 0) {
      writeSimulation(pd);
    }
  }
  return 0;
}

Parameters readParameters(std::string datamain) {
  InputFile in(datamain);
  Parameters par;
  in.getOption("dt", InputFile::Required) >> par.dt;
  in.getOption("temperature", InputFile::Required) >> par.temperature;
  in.getOption("lbox", InputFile::Required) >> par.lbox;
  in.getOption("simulationTime", InputFile::Required) >> par.simulationTime;
  in.getOption("printTime", InputFile::Required) >> par.printTime;
  in.getOption("initPos", InputFile::Required) >> par.initpos;
  in.getOption("bondFile", InputFile::Required) >> par.bondfile;
  return par;
}

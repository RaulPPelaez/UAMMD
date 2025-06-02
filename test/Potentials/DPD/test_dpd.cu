#include "Integrator/VerletNVE.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/Potential/DPD.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace uammd;
using std::endl;
using std::make_shared;
using NeighbourList = VerletList;

struct Parameters {
  real dt = 0.01;           // Time step
  real L = 10.0;            // Box size
  real cutOff_dpd = 1.0;    // Cut-off distance for DPD interactions
  real A_dpd = 25.0;        // Amplitude of the DPD force
  real gamma_dpd = 1.0;     // Damping coefficient for DPD
  real temperature = 1.0;   // Temperature for the system
  real gamma_par_dpd = 1.0; // Parallel damping coefficient for transversal DPD
  real gamma_perp_dpd =
      1.0; // Perpendicular damping coefficient for transversal DPD
};

void initVelocities(std::shared_ptr<ParticleData> pd, real temperature) {
  // Set all velocities to random numbers with the boltzmann distribution
  auto vel = pd->getVel(access::location::cpu, access::mode::write);
  std::mt19937 gen(pd->getSystem()->rng().next());
  real mean = 0;
  real stdev = sqrt(temperature);
  std::normal_distribution<real> dis(mean, stdev);
  std::generate(vel.begin(), vel.end(),
                [&]() { return make_real3(dis(gen), dis(gen), dis(gen)); });
}

auto createIntegratorTransversalDPD(std::shared_ptr<ParticleData> pd,
                                    const Parameters &ipar) {
  using NVE = VerletNVE;
  NVE::Parameters par;
  par.dt = ipar.dt;
  par.initVelocities = false;
  initVelocities(pd, ipar.temperature);
  auto verlet = std::make_shared<NVE>(pd, par);
  using DPD = Potential::DPD_impl<Potential::TransversalDissipation>;
  using DPDPF = PairForces<DPD, NeighbourList>;
  DPD::Parameters dpd_params;
  auto gamma = std::make_shared<Potential::TransversalDissipation>(
      ipar.gamma_par_dpd, ipar.gamma_perp_dpd, ipar.temperature, ipar.dt);
  dpd_params.cutOff = ipar.cutOff_dpd;
  dpd_params.gamma = gamma;
  dpd_params.dt = par.dt;
  auto pot = std::make_shared<DPD>(dpd_params);
  DPDPF::Parameters params;
  params.box = Box(ipar.L);
  auto pairforces = std::make_shared<DPDPF>(pd, params, pot);
  verlet->addInteractor(pairforces);
  return verlet;
}

auto createIntegratorDPD(std::shared_ptr<ParticleData> pd,
                         const Parameters &ipar) {
  using NVE = VerletNVE;
  NVE::Parameters par;
  par.dt = ipar.dt;
  par.initVelocities = false;
  initVelocities(pd, ipar.temperature);
  auto verlet = std::make_shared<NVE>(pd, par);
  using DPD = PairForces<Potential::DPD, NeighbourList>;
  Potential::DPD::Parameters dpd_params;
  auto gamma = std::make_shared<Potential::DefaultDissipation>(
      ipar.A_dpd, ipar.gamma_dpd, ipar.temperature, ipar.dt);
  dpd_params.cutOff = ipar.cutOff_dpd;
  dpd_params.gamma = gamma;
  dpd_params.dt = par.dt;
  auto pot = std::make_shared<Potential::DPD>(dpd_params);
  DPD::Parameters params;
  params.box = Box(ipar.L);
  auto pairforces = std::make_shared<DPD>(pd, params, pot);
  verlet->addInteractor(pairforces);
  return verlet;
}

enum class DissipationType { Default, Transversal };

std::shared_ptr<Integrator> createIntegrator(std::shared_ptr<ParticleData> pd,
                                             const Parameters &ipar,
                                             DissipationType type) {
  std::shared_ptr<Integrator> verlet;
  if (type == DissipationType::Transversal) {
    verlet = createIntegratorTransversalDPD(pd, ipar);
  } else if (type == DissipationType::Default) {
    verlet = createIntegratorDPD(pd, ipar);
  } else {
    throw std::invalid_argument("Unknown dissipation type");
  }
  return verlet;
}

real4 runSimulation(const Parameters &ipar, real r) {
  int N = 2;
  auto pd = make_shared<ParticleData>(N);
  Box box(ipar.L);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<real> dis(-0.5 * ipar.L, 0.5 * ipar.L);
  {
    auto pos = pd->getPos(access::cpu, access::write);
    auto ori = make_real4(make_real3(dis(gen), dis(gen), dis(gen)), 0);
    pos[0] = make_real4(-r * 0.5, 0, 0, 0) + ori;
    pos[1] = make_real4(r * 0.5, 0, 0, 0) + ori;
  }
  auto dpd = createIntegratorDPD(pd, ipar);
  dpd->forwardTime();
  real4 pos0 = pd->getPos(access::cpu, access::read)[0];
  return pos0;
}

void setPositionsInCubicBox(std::shared_ptr<ParticleData> pd, real L) {
  auto pos = pd->getPos(access::cpu, access::write);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<real> dis(-0.5 * L, 0.5 * L);
  for (auto &p : pos) {
    p = make_real4(dis(gen), dis(gen), dis(gen), 0);
  }
}

class DPDTest : public ::testing::TestWithParam<DissipationType> {};

TEST_P(DPDTest, SingleSimulationTest) {
  Parameters ipar;
  ipar.dt = 0.01;         // Time step
  ipar.L = 32.0;          // Box size
  ipar.cutOff_dpd = 1.0;  // Cut-off distance for DPD interactions
  ipar.A_dpd = 25.0;      // Amplitude of the DPD force
  ipar.gamma_dpd = 1.0;   // Damping coefficient for DPD
  ipar.temperature = 0.0; // Temperature for the system
  // Set the distance between particles
  real r = 1.0; // Distance between particles
  real4 pos0 = runSimulation(ipar, r);
}

TEST_P(DPDTest, MomentumIsConservedTest) {
  Parameters ipar;

  ipar.L = 32.0;          // Box size
  ipar.cutOff_dpd = 1.0;  // Cut-off distance for DPD interactions
  ipar.A_dpd = 25.0;      // Amplitude of the DPD force
  ipar.gamma_dpd = 1.0;   // Damping coefficient for DPD
  ipar.temperature = 1.0; // Temperature for the system
  real boltzmannVelocityAmplitude = sqrt(3 * ipar.temperature);
  ipar.dt = 0.01 * ipar.cutOff_dpd / boltzmannVelocityAmplitude;
  int N = 10000;
  auto pd = make_shared<ParticleData>(N);
  setPositionsInCubicBox(pd, ipar.L);
  auto dpd = createIntegratorDPD(pd, ipar);
  int nsteps = 100;
  for (int i = 0; i < nsteps; ++i) {
    dpd->forwardTime();
    {
      auto force = pd->getForce(access::cpu, access::read);
      auto ftot =
          make_real3(std::accumulate(force.begin(), force.end(), real4{})) / N;
      EXPECT_NEAR(ftot.x, 0, 1e-6);
      EXPECT_NEAR(ftot.y, 0, 1e-6);
      EXPECT_NEAR(ftot.z, 0, 1e-6);
    }
  }
}

TEST_P(DPDTest, TemperatureTest) {
  // Measure the temperature of a system and check if it is close to the
  // expected value.
  Parameters ipar;
  ipar.L = 32.0;             // Box size
  ipar.cutOff_dpd = 2.0;     // Cut-off distance for DPD interactions
  ipar.A_dpd = 25.0;         // Amplitude of the DPD force
  ipar.gamma_dpd = 17.0;     // Damping coefficient for DPD
  ipar.gamma_par_dpd = 17.0; // Parallel damping coefficient for transversal DPD
  ipar.gamma_perp_dpd =
      17.0; // Perpendicular damping coefficient for transversal DPD
  ipar.temperature = 0.921321; // Temperature for the system
  real boltzmannVelocityAmplitude = sqrt(ipar.temperature);
  real charTime = ipar.cutOff_dpd / boltzmannVelocityAmplitude;
  ipar.dt =
      0.005 *
      charTime; // Set the time step to be a fraction of the characteristic time
  int N = 10000;
  auto pd = make_shared<ParticleData>(N);
  setPositionsInCubicBox(pd, ipar.L);
  auto dpd = createIntegratorDPD(pd, ipar);
  int isteps =
      100 * charTime / ipar.dt; // Run for a few characteristic times to
  // allow the system to equilibrate
  auto computeTemperature = [&pd]() {
    auto vel = pd->getVel(access::cpu, access::read);
    real kineticEnergy = 0.0;
    for (const auto &v : vel) {
      kineticEnergy += 0.5 * dot(v, v);
    }
    return (2.0 / 3.0) * kineticEnergy / pd->getNumParticles();
  };
  for (int i = 0; i < isteps; ++i) {
    dpd->forwardTime();
  }
  int nsteps = 100 * charTime / ipar.dt;
  std::vector<double> temperatures;
  for (int i = 0; i < nsteps; ++i) {
    dpd->forwardTime();
    const auto t = computeTemperature();
    temperatures.push_back(t);
  }
  double temperature =
      std::accumulate(temperatures.begin(), temperatures.end(), 0.0) /
      temperatures.size();
  // Error depens on dt, the lower the dt, the lower the error
  EXPECT_NEAR(temperature, ipar.temperature, 0.01 * ipar.temperature)
      << "Expected temperature: " << ipar.temperature
      << ", measured temperature: " << temperature;
}

std::string
DissipationTypeToString(const ::testing::TestParamInfo<DissipationType> &info) {
  switch (info.param) {
  case DissipationType::Default:
    return "Default";
  case DissipationType::Transversal:
    return "Transversal";
  default:
    return "Unknown";
  }
}

INSTANTIATE_TEST_SUITE_P(DissipationModes, DPDTest,
                         ::testing::Values(DissipationType::Default,
                                           DissipationType::Transversal),
                         DissipationTypeToString);

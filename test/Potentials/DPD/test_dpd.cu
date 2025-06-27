#include "Integrator/VerletDPD.cuh"
#include "Interactor/ExternalForces.cuh"
#include "msd/msd.hpp"
#include <algorithm>
#include <cmath>
#include <execution>
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>
using namespace uammd;
using std::endl;
using std::make_shared;

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
  using Verlet = dpd::Verlet<dpd::TransversalDissipation>;
  Verlet::Parameters par;
  par.dt = ipar.dt;
  par.is2D = false;
  par.mass = 1;
  par.dissipation = std::make_shared<dpd::TransversalDissipation>(
      ipar.gamma_par_dpd, ipar.gamma_perp_dpd, ipar.A_dpd, ipar.temperature,
      ipar.dt);
  par.rcut = ipar.cutOff_dpd;
  par.box = Box(ipar.L);
  par.temperature = ipar.temperature;
  par.lambda = 0.65;
  auto verlet = std::make_shared<Verlet>(pd, par);
  return verlet;
}

auto createIntegratorDPD(std::shared_ptr<ParticleData> pd,
                         const Parameters &ipar) {
  using Verlet = dpd::Verlet<dpd::DefaultDissipation>;
  Verlet::Parameters par;
  par.dt = ipar.dt;
  par.is2D = false;
  par.mass = 1;
  par.temperature = ipar.temperature;
  par.dissipation = std::make_shared<dpd::DefaultDissipation>(
      ipar.A_dpd, ipar.gamma_dpd, ipar.temperature, ipar.dt);
  par.rcut = ipar.cutOff_dpd;
  par.box = Box(ipar.L);
  par.lambda = 0.65;
  auto verlet = std::make_shared<Verlet>(pd, par);

  return verlet;
}

enum class DissipationType { Default, Transversal };

std::string
DissipationTypeToString(::testing::TestParamInfo<DissipationType> utype) {
  DissipationType type = utype.param;
  switch (type) {
  case DissipationType::Default:
    return "Default";
  case DissipationType::Transversal:
    return "Transversal";
  default:
    return "Unknown";
  }
}

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
#define DPD_PARAM GetParam()

TEST_P(DPDTest, MomentumIsConservedTest) {
  Parameters ipar;

  ipar.L = 32.0;            // Box size
  ipar.cutOff_dpd = 1.0;    // Cut-off distance for DPD interactions
  ipar.A_dpd = 25.0;        // Amplitude of the DPD force
  ipar.gamma_dpd = 1.0;     // Damping coefficient for DPD
  ipar.gamma_par_dpd = 1.0; // Parallel damping coefficient for transversal DPD
  ipar.gamma_perp_dpd =
      0.000000000001; // Perpendicular damping coefficient for transversal DPD
  ipar.temperature = 1.0; // Temperature for the system
  real boltzmannVelocityAmplitude = sqrt(3 * ipar.temperature);
  ipar.dt = 0.01 * ipar.cutOff_dpd / boltzmannVelocityAmplitude;
  int N = 10000;
  auto pd = make_shared<ParticleData>(N);
  setPositionsInCubicBox(pd, ipar.L);
  auto dpd = createIntegrator(pd, ipar, DPD_PARAM);
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
  ipar.L = 32.0;         // Box size
  ipar.cutOff_dpd = 2.0; // Cut-off distance for DPD interactions
  ipar.A_dpd = 25.0;     // Amplitude of the DPD force
  ipar.gamma_dpd = 1.0;  // Damping coefficient for DPD
  ipar.gamma_par_dpd =
      ipar.gamma_dpd; // Parallel damping coefficient for transversal DPD
  ipar.gamma_perp_dpd =
      0.000000001; // Perpendicular damping coefficient for transversal DPD
  ipar.temperature = 0.921321; // Temperature for the system
  real boltzmannVelocityAmplitude = sqrt(ipar.temperature);
  real charTime = ipar.cutOff_dpd / boltzmannVelocityAmplitude;
  ipar.dt =
      0.001 *
      charTime; // Set the time step to be a fraction of the characteristic time
  int N = 10000;
  auto pd = make_shared<ParticleData>(N);
  setPositionsInCubicBox(pd, ipar.L);
  auto dpd = createIntegrator(pd, ipar, DPD_PARAM);
  int isteps = 10 * charTime / ipar.dt; // Run for a few characteristic times to
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
  int nsteps = std::max(int(1 * charTime / ipar.dt), 1000);
  int nprint = nsteps / 1000;
  nsteps = (nsteps / nprint) * nprint; // Ensure nsteps is a multiple of nprint
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

std::vector<double>
computeVACF(const std::vector<std::vector<real3>> &velocities) {
  int Nsteps = velocities.size();
  int Nparticles = velocities[0].size();
  std::vector<double> vacf(Nsteps, 0.0);

  // Compute VACF(t) = ⟨v_i(t+t0) · v_i(t0)⟩ averaged over all i and all t0
  for (int t0 = 0; t0 < Nsteps; ++t0) {
    // Counting iterator with ranges
    auto cit = std::ranges::iota_view{0, Nsteps - t0};
    std::for_each(std::execution::par_unseq, cit.begin(), cit.end(),
                  [&](int t) {
                    //    for (int t = 0; t < Nsteps - t0; ++t) {
                    double sum = 0.0;
                    for (int i = 0; i < Nparticles; ++i) {
                      sum += dot(velocities[t + t0][i], velocities[t0][i]);
                    }
                    vacf[t] += sum / Nparticles;
                  });
  }
  // Normalize by the number of time steps
  for (int t = 0; t < Nsteps; ++t) {
    vacf[t] /= (Nsteps - t);
  }
  return vacf;
}

auto fit_exponential_decay(std::span<double> vacf) {
  // We assume the form: A * exp(-t / tau)
  // Linearize: log(VACF) = log(A) - t / tau
  if (vacf.size() < 2) {
    throw std::invalid_argument("Not enough data points to fit an exponential");
  }
  std::vector<double> t;
  std::vector<double> y;
  for (size_t i = 0; i < vacf.size(); ++i) {
    if (vacf[i] <= 0.0)
      continue; // skip non-positive values to avoid log issues
    t.push_back(static_cast<double>(i));
    y.push_back(std::log(vacf[i]));
  }

  if (t.size() < 2) {
    throw std::runtime_error("Not enough positive VACF values to perform fit");
  }
  double mean_t = std::accumulate(t.begin(), t.end(), 0.0) / t.size();
  double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t i = 0; i < t.size(); ++i) {
    double dt = t[i] - mean_t;
    double dy = y[i] - mean_y;
    numerator += dt * dy;
    denominator += dt * dt;
  }
  double slope = numerator / denominator;
  double intercept = mean_y - slope * mean_t;
  double tau = -1.0 / slope;
  double A = std::exp(intercept);
  return std::make_pair(A, tau); // return as (A, tau)
}

TEST_P(DPDTest, VelocityAutocorrelationTest) {
  Parameters ipar;
  ipar.L = 32.0;
  ipar.cutOff_dpd = 2.0;
  ipar.A_dpd = 0.0;
  ipar.gamma_dpd = 0.1;
  ipar.gamma_par_dpd = ipar.gamma_dpd;
  ipar.gamma_perp_dpd = 0.0000001;
  // ipar.temperature must be much larger than pow(ipar.gamma_dpd/3.0, 2) for
  // the VACF to be a single exponential decay
  ipar.temperature = 100.0 * pow(ipar.gamma_dpd / 3.0, 2);
  real time_thermal = ipar.cutOff_dpd / sqrt(ipar.temperature);
  real decay_t = 3.0 / (ipar.gamma_dpd);
  real charTime = std::min(time_thermal, decay_t);
  ipar.dt = 0.005 * charTime;

  int N = 10000;
  auto pd = make_shared<ParticleData>(N);
  setPositionsInCubicBox(pd, ipar.L);
  auto dpd = createIntegrator(pd, ipar, DPD_PARAM);

  // Equilibrate system
  const int equilibrationSteps = 1 * charTime / ipar.dt;
  for (int i = 0; i < equilibrationSteps; ++i)
    dpd->forwardTime();

  // Record velocities over time
  int nsteps = std::max(int(2 * decay_t / ipar.dt), 1500);
  int nprint = nsteps / 1500;
  nsteps = (nsteps / nprint) * nprint; // Ensure nsteps is a multiple of nprint
  std::vector<std::vector<real3>> velocities;
  for (int step = 0; step < nsteps; ++step) {
    dpd->forwardTime();
    if (step % nprint != 0)
      continue; // Skip steps not in the print interval
    {
      auto vel = pd->getVel(access::cpu, access::read);
      velocities.emplace_back();
      velocities.back().reserve(N);
      for (const auto &v : vel) {
        velocities.back().push_back(make_real3(v));
      }
    }
  }

  std::vector<double> vacf = computeVACF(velocities);
  // {
  //   // Write to a file
  //   std::string name =
  //       GetParam() == DissipationType::Default ? "default" : "transversal";
  //   std::ofstream vacf_file("vacf" + name + ".dat");
  //   std::cerr << "dt is " << ipar.dt << std::endl;
  //   for (int t = 0; t < vacf.size(); ++t) {
  //     vacf_file << t * ipar.dt * nprint / decay_t << " "
  //               << vacf[t] / (3.0 * ipar.temperature) << endl;
  //   }
  // }

  EXPECT_NEAR(vacf[0], 3.0 * ipar.temperature, 0.1)
      << "VACF at t=0 should be close to 3.0";

  // Get the values until the vacf drops to 0.8
  auto it = std::find_if(vacf.begin(), vacf.end(), [&](double v) {
    return v < 0.8 * 3.0 * ipar.temperature;
  });
  if (it == vacf.end()) {
    FAIL() << "VACF did not drop below 0.8 * 3.0 * temperature";
  }
  int t_end = std::distance(vacf.begin(), it);
  // Fit to an exponential decay
  std::span vacf_span(vacf.begin(), t_end);
  auto [pref, tau] = fit_exponential_decay(vacf_span);
  auto theory_tau = decay_t / (ipar.dt * nprint);
  EXPECT_NEAR(tau, theory_tau, 0.1 * theory_tau)
      << "Fitted decay time should be close to the theoretical value";
}

std::vector<double>
computeMSD(const std::vector<std::vector<real3>> &positions) {
  int Nsteps = positions.size();
  int Nparticles = positions[0].size();
  std::vector<double> msd(Nsteps, 0.0);

  for (int t = 0; t < Nsteps; ++t) {
    double sum = 0.0;
    for (int i = 0; i < Nparticles; ++i) {
      real3 dr = positions[t][i] - positions[0][i];
      sum += dot(dr, dr);
    }
    msd[t] = sum / Nparticles;
  }

  return msd;
}

template <typename T> T fit_to_line(std::span<T> vec) {
  // Fit a line to the data in vec using least squares
  // Returns the slope of the line
  int n = vec.size();
  if (n < 2) {
    throw std::invalid_argument("Not enough data points to fit a line");
  }
  T sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
  for (int i = 0; i < n; ++i) {
    sum_x += i;
    sum_y += vec[i];
    sum_xy += i * vec[i];
    sum_xx += i * i;
  }
  T slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
  return slope;
}

TEST_P(DPDTest, DiffusionCoefficientTest) {
  Parameters ipar;
  ipar.L = 32.0;
  ipar.cutOff_dpd = 2.0;
  ipar.A_dpd = 25.0;
  ipar.gamma_dpd = 170.0;
  ipar.gamma_par_dpd = 170.0;
  ipar.gamma_perp_dpd = 0.00000001;
  ipar.temperature = 0.921321;
  real boltzmannVelocityAmplitude = sqrt(ipar.temperature);
  real charTime = ipar.cutOff_dpd / boltzmannVelocityAmplitude;
  ipar.dt = 0.005 * charTime;
  int N = 10000;
  auto pd = make_shared<ParticleData>(N);
  setPositionsInCubicBox(pd, ipar.L);
  auto dpd = createIntegrator(pd, ipar, DPD_PARAM);

  // Equilibrate
  int isteps = 5 * charTime / ipar.dt; // Equilibration steps
  for (int i = 0; i < isteps; ++i)
    dpd->forwardTime();

  // Record positions over time
  int nsteps = 300 * charTime / ipar.dt; // Total steps to record
  int nprint = nsteps / 3000;            // Print every 100th step
  nsteps = (nsteps / nprint) * nprint; // Ensure nsteps is a multiple of nprint
  std::vector<double> positions; // Stores position such that particle i, time t
  // and dimension j is located at
  // positions[t + signal_size * (3 * i + k)]
  int ntimes = (nsteps / nprint);
  positions.resize(ntimes * 3 * N);
  int iprint = 0;
  for (int step = 0; step < nsteps; ++step) {
    dpd->forwardTime();
    if (step % nprint != 0)
      continue; // Skip steps not in the print interval
    auto pos = pd->getPos(access::cpu, access::read);
    for (int i = 0; i < N; ++i) {
      real3 p = make_real3(pos[i]);
      // Check that there are not NaN values in the positions
      if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) {
        throw std::runtime_error("NaN value found in particle position");
      }
      positions[iprint + ntimes * (3 * i + 0)] = p.x;
      positions[iprint + ntimes * (3 * i + 1)] = p.y;
      positions[iprint + ntimes * (3 * i + 2)] = p.z;
    }
    iprint++;
  }
  ASSERT_EQ(iprint, ntimes)
      << "Number of printed steps does not match expected number";
  auto msd = msd::mean_square_displacement(std::span(positions),
                                           msd::device::cpu, N, ntimes, 3);
  // Estimate D from slope of MSD in linear regime
  // Start at 10*charTime and go up to 100*charTime
  int t_start = 10 * charTime / ipar.dt / nprint;
  int t_end = 100 * charTime / ipar.dt / nprint;
  std::span msd_x(msd);
  msd_x = msd_x.subspan(t_start, t_end - t_start);
  double slopex = fit_to_line(msd_x);
  std::span msd_y(msd.begin() + ntimes, ntimes);
  msd_y = msd_y.subspan(t_start, t_end - t_start);
  double slopey = fit_to_line(msd_y);
  std::span msd_z(msd.begin() + 2 * ntimes, ntimes);
  msd_z = msd_z.subspan(t_start, t_end - t_start);
  double slopez = fit_to_line(msd_z);
  // MSD(t) = 2 * D * t // In each dimension
  real3 D = make_real3(slopex, slopey, slopez) / (2.0 * ipar.dt * nprint);
  // Check that all three are similar
  EXPECT_NEAR(D.x, D.y, 0.1 * D.x)
      << "Diffusion coefficients in x and y should be similar";
  EXPECT_NEAR(D.x, D.z, 0.1 * D.x)
      << "Diffusion coefficients in x and z should be similar";
  // // Write MSD to file for analysis
  // std::string name =
  //     GetParam() == DissipationType::Default ? "default" : "transversal";
  // std::ofstream msd_file("msd" + name + ".dat");
  // for (int t = 0; t < ntimes; ++t) {
  //   msd_file << t * ipar.dt * nprint;
  //   for (int i = 0; i < 3; ++i) {
  //     msd_file << " " << msd[t + ntimes * i];
  //   }
  //   msd_file << endl;
  // }
  // msd_file.close();
  EXPECT_GT(D.x, 0.0);
  EXPECT_GT(D.y, 0.0);
  EXPECT_GT(D.z, 0.0);

  real Dtot = (D.x + D.y + D.z) / 3.0;
  real Dtheo = ipar.temperature / (3.0 * ipar.gamma_dpd);
  std::cerr << "Dtot = " << Dtot << ", Dtheo = " << Dtheo << std::endl;
  EXPECT_NEAR(Dtot, Dtheo, 0.1 * Dtheo)
      << "Diffusion coefficient should be close to the theoretical value";
}

struct SinusoidalForce {
  real amplitude = 1.0; // Amplitude of the sinusoidal force
  real L;
  __device__ ForceEnergyVirial sum(Interactor::Computables comp, real4 pos) {
    real3 f{};
    // Add a force in the x-direction that oscillates sinusoidally in the
    // y-direction The force should be periodic
    f.y = amplitude * sin(2 * M_PI * pos.x / L);
    real energy = comp.energy ? 0 : 0;
    real virial = comp.virial ? 0 : 0;
    return {f, energy, virial};
  }

  auto getArrays(ParticleData *pd) {
    auto pos = pd->getPos(access::gpu, access::read);
    return pos.begin();
  }
};

TEST_P(DPDTest, ShearViscosityTest) {
  // Add a sinusoidal external forcing (must be periodic), leave the system
  // equilibrate and measure the terminal velocity of the particles. The shear
  // viscosity is then given by the ratio of the force amplitude to the terminal
  // velocity.
  Parameters ipar;
  ipar.L = 32.0;          // Box size
  ipar.cutOff_dpd = 2.0;  // Cut-off distance for DPD interactions
  ipar.A_dpd = 0.0;       // Amplitude of the DPD force
  ipar.gamma_dpd = 100.0; // Damping coefficient for DPD
  ipar.gamma_par_dpd =
      ipar.gamma_dpd; // Parallel damping coefficient for transversal DPD
  // Perpendicular damping coefficient for transversal DPD
  ipar.gamma_perp_dpd = 0.0000001;
  ipar.temperature = 0.0; // Temperature for the system
  real charTime = 1.0 / ipar.gamma_dpd;
  ipar.dt = 0.005 * charTime; // Set the time step to be a fraction of the
  const real density = 0.3;
  int N = pow(ipar.L, 3.0) / density;
  auto pd = make_shared<ParticleData>(N);
  setPositionsInCubicBox(pd, ipar.L);
  auto dpd = createIntegrator(pd, ipar, DPD_PARAM);
  auto gr = std::make_shared<SinusoidalForce>();
  gr->amplitude = 1.0; // Amplitude of the sinusoidal force
  gr->L = ipar.L;      // Set the box size in the sinusoidal force
  auto ext = std::make_shared<ExternalForces<SinusoidalForce>>(pd, gr);
  dpd->addInteractor(ext);
  // Equilibrate the system
  int isteps = 20 * charTime / ipar.dt; // Equilibration steps
  for (int i = 0; i < isteps; ++i)
    dpd->forwardTime();
  // Get velocities and average them in the Z and Y directions
  int nsim = std::max(int(2 * charTime / ipar.dt),
                      1000); // Number of steps to average over
  int nprint = nsim / 1000;
  nsim = (nsim / nprint) * nprint; // Ensure nsim is a multiple of navg
  // Make an histogram of velocities in the x-direction
  int n_bins = 300;
  std::vector<real> vel_avg(n_bins, 0.0);
  std::vector<int> bin_count(n_bins, 0);
  for (int i = 0; i < nsim; ++i) {
    dpd->forwardTime();
    if (i % nprint != 0)
      continue;
    auto vel = pd->getVel(access::cpu, access::read);
    auto pos = pd->getPos(access::cpu, access::read);
    for (int i = 0; i < N; ++i) {
      // bin depends on the position of the particle in the box in the
      // x-direction using MIC Box goes from -0.5L to 0.5L
      real p_in_box = pos[i].x - ipar.L * floor(pos[i].x / ipar.L + 0.5);
      // Tkae it to the range [0, L)
      p_in_box += ipar.L * 0.5;
      int bin = static_cast<int>(p_in_box / ipar.L * n_bins);
      if (bin >= 0 && bin < n_bins) {
        vel_avg[bin] += vel[i].y;
        bin_count[bin]++;
      }
    }
  }
  // Print them all to a file
  std::string name =
      GetParam() == DissipationType::Default ? "default" : "transversal";
  std::ofstream vel_file("velocities" + name + ".dat");
  for (int i = 0; i < n_bins; ++i) {
    if (bin_count[i] > 0) {
      vel_avg[i] /= bin_count[i]; // Average the velocities in the bin
    }
    vel_file << i * ipar.L / n_bins << " " << vel_avg[i] << endl;
  }

  constexpr real s = 2.0; // Exponent of the DPD dissipative kernel
  real vis_theo =
      2 * M_PI * ipar.gamma_dpd * density * density * pow(ipar.cutOff_dpd, 5) /
      15 *
      (1 / (s + 1) - 4 / (s + 2) + 6 / (s + 3) - 4 / (s + 4) + 1 / (s + 5));
  real v_amp = (*std::max_element(vel_avg.begin(), vel_avg.end()) -
                *std::min_element(vel_avg.begin(), vel_avg.end())) *
               0.5;
  real viscosity_measured = density * ipar.L / (pow(2 * M_PI, 2) * v_amp);
  EXPECT_NEAR(viscosity_measured, vis_theo, 0.1 * vis_theo)
      << "Shear viscosity should be close to the theoretical value";
}

INSTANTIATE_TEST_SUITE_P(DissipationModes, DPDTest,
                         ::testing::Values(DissipationType::Default,
                                           DissipationType::Transversal),
                         DissipationTypeToString);

/*Raul P. Pelaez 2022. Tests for the DPStokes algorithm.

The DPStokes algorithm can be accessed as an independent mobility solver and as
an UAMMD Integrator. These tests first stablish the correctness of the solver
and then the integrator. The latter is built upon the former and adds
fluctuations to it.
 */
#include "Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"
#include "gmock/gmock.h"
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <random>

using namespace uammd;
using DPStokesSlab_ns::DPStokes;
using DPStokesSlab_ns::DPStokesIntegrator;

using DPStokesSlab_ns::cached_vector;

TEST(DPStokes, CanBeCreated) {
  DPStokes::Parameters par;
  par.nx = 32;
  par.ny = 32;
  par.nz = 32;
  par.w = 6;
  par.beta.x = 13;
  par.alpha = par.w * 0.5;
  par.mode = DPStokes::WallMode::slit;
  par.viscosity = 1;
  par.Lx = par.Ly = 1;
  par.H = 1;
  par.tolerance = 1e-4;
  auto dpstokes = std::make_shared<DPStokes>(par);
}

// Paramters for a support of w=6
auto getDPStokesParamtersOnlyForce(real Lx, real Ly, real H, real viscosity,
                                   real hydrodynamicRadius) {
  real h = hydrodynamicRadius / 1.554;
  int nx = int(Lx / h + 0.5);
  int ny = int(Ly / h + 0.5);
  DPStokes::Parameters par;
  par.nx = nx;
  par.ny = ny;
  par.nz = int(M_PI * H / (2 * h));
  par.w = 6;
  par.beta.x = 1.714 * par.w;
  par.alpha = par.w * 0.5;
  par.mode = DPStokes::WallMode::slit;
  par.viscosity = viscosity;
  par.Lx = Lx;
  par.Ly = Ly;
  par.H = H;
  par.tolerance = 1e-4;
  return par;
}
auto getDPStokesParamtersOnlyForce(real Lxy, real H, real viscosity,
                                   real hydrodynamicRadius) {
  return getDPStokesParamtersOnlyForce(Lxy, Lxy, H, viscosity,
                                       hydrodynamicRadius);
}

// Check that a force pulling from a single particle only moves it in that
// direction
TEST(DPStokes, PulledParticleDoesNotMoveInOtherDirection) {
  real hydrodynamicRadius = 1;
  real Lxy = 32;
  real H = Lxy;
  auto par =
      getDPStokesParamtersOnlyForce(Lxy, H, 1 / (6 * M_PI), hydrodynamicRadius);
  par.mode = DPStokes::WallMode::slit;
  auto dpstokes = std::make_shared<DPStokes>(par);
  cached_vector<real3> pos(1);
  pos[0] = {0, 0, 0};
  cached_vector<real3> force(1);
  force[0] = {1, 0, 0};
  auto mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  real3 disp = mf[0];
  ASSERT_THAT(disp.x, ::testing::Not(::testing::DoubleNear(0.0, 1e-13)));
  ASSERT_THAT(disp.y, ::testing::DoubleNear(0.0, 1e-13));
  ASSERT_THAT(disp.z, ::testing::DoubleNear(0.0, 1e-13));
  force[0] = {0, 1, 0};
  mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  disp = mf[0];
  ASSERT_THAT(disp.y, ::testing::Not(::testing::DoubleNear(0.0, 1e-13)));
  ASSERT_THAT(disp.x, ::testing::DoubleNear(0.0, 1e-13));
  ASSERT_THAT(disp.z, ::testing::DoubleNear(0.0, 1e-13));
  force[0] = {0, 0, 1};
  mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  disp = mf[0];
  ASSERT_THAT(disp.z, ::testing::Not(::testing::DoubleNear(0.0, 1e-13)));
  ASSERT_THAT(disp.y, ::testing::DoubleNear(0.0, 1e-13));
  ASSERT_THAT(disp.x, ::testing::DoubleNear(0.0, 1e-13));
}

// The self mobility should converge towards the open boundary case as
// M0(L)\approx M0(1+a/L) Where "a" is an constant around 1. The DPStokes
// algorithm only guarantees about 4 digits of accuracy. So a box size of L=128
// rh
//   should yield 2-3 digits around M0
// Here we measure self mobility by pulling from a single particle
// This should happen in any geometry
TEST(DPStokes, ReproducesOpenBoundarySelfMobilityWithLargeDomainSlitChannel) {
  real hydrodynamicRadius = 1;
  real Lxy = 128;
  real H = Lxy;
  auto par =
      getDPStokesParamtersOnlyForce(Lxy, H, 1 / (6 * M_PI), hydrodynamicRadius);
  par.mode = DPStokes::WallMode::slit;
  auto dpstokes = std::make_shared<DPStokes>(par);
  cached_vector<real3> pos(1);
  pos[0] = {0, 0, 0};
  cached_vector<real3> force(1);
  force[0] = {1, 1, 1};
  auto mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  real3 disp = mf[0];
  ASSERT_THAT(disp.x, ::testing::DoubleNear(1.0, 0.1));
  ASSERT_THAT(disp.y, ::testing::DoubleNear(1.0, 0.1));
  ASSERT_THAT(disp.z, ::testing::DoubleNear(1.0, 0.1));
}

TEST(DPStokes, ReproducesOpenBoundarySelfMobilityWithLargeDomainBottomWall) {
  real hydrodynamicRadius = 1;
  real Lxy = 128;
  real H = Lxy;
  auto par =
      getDPStokesParamtersOnlyForce(Lxy, H, 1 / (6 * M_PI), hydrodynamicRadius);
  par.mode = DPStokes::WallMode::bottom;
  auto dpstokes = std::make_shared<DPStokes>(par);
  cached_vector<real3> pos(1);
  pos[0] = {0, 0, 0};
  cached_vector<real3> force(1);
  force[0] = {1, 1, 1};
  auto mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  real3 disp = mf[0];
  ASSERT_THAT(disp.x, ::testing::DoubleNear(1.0, 0.1));
  ASSERT_THAT(disp.y, ::testing::DoubleNear(1.0, 0.1));
  ASSERT_THAT(disp.z, ::testing::DoubleNear(1.0, 0.1));
}

TEST(DPStokes, ReproducesOpenBoundarySelfMobilityWithLargeDomainOpen) {
  real hydrodynamicRadius = 1;
  real Lxy = 128;
  real H = Lxy * 0.25;
  auto par =
      getDPStokesParamtersOnlyForce(Lxy, H, 1 / (6 * M_PI), hydrodynamicRadius);
  par.mode = DPStokes::WallMode::none;
  auto dpstokes = std::make_shared<DPStokes>(par);
  cached_vector<real3> pos(1);
  pos[0] = {0, 0, 0};
  cached_vector<real3> force(1);
  force[0] = {1, 1, 1};
  auto mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  real3 disp = mf[0];
  ASSERT_THAT(disp.x, ::testing::DoubleNear(1.0, 0.1));
  ASSERT_THAT(disp.y, ::testing::DoubleNear(1.0, 0.1));
  ASSERT_THAT(disp.z, ::testing::DoubleNear(1.0, 0.1));
}

// TESTS for the Integrator start here

auto getDPStokesIntegratorParamtersOnlyForce(real Lxy, real H, real viscosity,
                                             real hydrodynamicRadius) {
  DPStokes::Parameters par =
      getDPStokesParamtersOnlyForce(Lxy, H, viscosity, hydrodynamicRadius);
  DPStokesIntegrator::Parameters pari;
  pari.nx = par.nx;
  pari.ny = par.ny;
  pari.nz = par.nz;
  pari.w = par.w;
  pari.beta = par.beta;
  pari.alpha = par.alpha;
  pari.mode = DPStokes::WallMode::slit;
  pari.viscosity = par.viscosity;
  pari.Lx = par.Lx;
  pari.Ly = par.Ly;
  pari.H = H;
  return pari;
}

struct miniInteractor : public Interactor {
public:
  miniInteractor(std::shared_ptr<ParticleData> pd) : Interactor(pd) {}
  void sum(Computables comp, cudaStream_t st = 0) override {
    pd->getForce(access::cpu, access::write)[0] = {1, 1, 1, 0};
  }
};

// Computes the same thing using the solver and the Integrator.
TEST(DPStokesIntegrator, ReproducesSelfMobilityOfSolverWithZeroTemperature) {
  real hydrodynamicRadius = 1;
  real Lxy = 32;
  real H = Lxy * 0.5;
  auto par =
      getDPStokesParamtersOnlyForce(Lxy, H, 1 / (6 * M_PI), hydrodynamicRadius);
  par.mode = DPStokes::WallMode::slit;
  auto dpstokes = std::make_shared<DPStokes>(par);
  cached_vector<real3> pos(1);
  pos[0] = {0, 0, 0};
  cached_vector<real3> force(1);
  force[0] = {1, 1, 1};
  auto mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  real3 disp = mf[0];
  auto pd = std::make_shared<ParticleData>(1);
  auto pari = getDPStokesIntegratorParamtersOnlyForce(Lxy, H, 1 / (6 * M_PI),
                                                      hydrodynamicRadius);
  pari.temperature = 0;
  pari.mode = par.mode;
  pari.dt = 1;
  pd->getPos(access::cpu, access::write)[0] = {0, 0, 0, 0};
  auto dpstokes_integrator = std::make_shared<DPStokesIntegrator>(pd, pari);
  dpstokes_integrator->addInteractor(std::make_shared<miniInteractor>(pd));
  dpstokes_integrator->forwardTime();
  auto mfi = pd->getPos(access::cpu, access::read)[0];
  ASSERT_THAT(mfi.x, ::testing::DoubleNear(disp.x, 1e-12));
  ASSERT_THAT(mfi.y, ::testing::DoubleNear(disp.y, 1e-12));
  ASSERT_THAT(mfi.z, ::testing::DoubleNear(disp.z, 1e-12));
}

// Ensures that the diffusion coeficient complies with fluctuation dissipation
// balance. First we place a single particle at a certain height without any
// forces and a low dt, so that over many realizations <(\Delta q)^2> =
// 2*temperature*M*dt Then we set the temperature to zero and we compute the
// mobility at each Z by pulling a single particle
//  so that over many realizations <\Delta q> = M*F*dt
//
TEST(DPStokesIntegrator, ObeysFluctuationDissipationBalanceAtEveryHeight) {
  real hydrodynamicRadius = 1.32232;
  real Lxy = 16 * hydrodynamicRadius;
  real H = 8 * hydrodynamicRadius;
  auto pd = std::make_shared<ParticleData>(1);
  auto pari = getDPStokesIntegratorParamtersOnlyForce(Lxy, H, 1 / (6 * M_PI),
                                                      hydrodynamicRadius);
  pari.temperature = 1.3452;
  pari.mode = DPStokes::WallMode::slit;
  pari.dt = 0.001;
  pd->getPos(access::cpu, access::write)[0] = {0, 0, 0, 0};
  auto dpstokes_integrator = std::make_shared<DPStokesIntegrator>(pd, pari);
  int nz = 10;
  std::vector<real3> d0(nz, real3());
  std::vector<real3> m0(nz, real3());
  for (int iz = 0; iz < nz; iz++) {
    real z = (-0.5 + (iz / real(nz - 1))) * (H - hydrodynamicRadius * 0.1);
    int navg = 10000;
    real3 avg = {0, 0, 0};
    for (int i = 0; i < navg; i++) {
      pd->getPos(access::cpu, access::write)[0] = {0, 0, z, 0};
      dpstokes_integrator->forwardTime();
      auto r = (make_real3(
          pd->getPos(access::cpu, access::read)[0]))-make_real3(0, 0, z);
      avg += r * r;
    }
    auto sigma = avg / navg;
    d0[iz] = sigma / (2 * pari.temperature * pari.dt);
  }
  pari.temperature = 0;
  dpstokes_integrator = std::make_shared<DPStokesIntegrator>(pd, pari);
  dpstokes_integrator->addInteractor(std::make_shared<miniInteractor>(pd));
  for (int iz = 0; iz < nz; iz++) {
    real z = (-0.5 + (iz / real(nz - 1))) * (H - hydrodynamicRadius * 0.1);
    pd->getPos(access::cpu, access::write)[0] = {0, 0, z, 0};
    dpstokes_integrator->forwardTime();
    auto r = (make_real3(pd->getPos(access::cpu, access::read)[0]));
    m0[iz] = (r - make_real3(0, 0, z)) / pari.dt;
    real3 err = abs((m0[iz] - d0[iz]) / m0[iz]);
    ASSERT_LE(err.x, 0.05);
    ASSERT_LE(err.y, 0.05);
    ASSERT_LE(err.z, 0.05);
  }
}

namespace duplicate {

auto genPositions(int N, real Lx, real Ly, real H) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<real> posDist(-0.5, 0.5);
  cached_vector<real3> pos(N);
  cached_vector<real3> force(N);
  for (int i = 0; i < N; ++i) {
    pos[i] = {posDist(gen) * Lx, posDist(gen) * Ly, posDist(gen) * H * 0.8};
    force[i] = {posDist(gen), posDist(gen), posDist(gen)};
  }
  // center forces
  real3 meanF{0, 0, 0};
  for (int i = 0; i < N; ++i) {
    real3 f = force[i];
    meanF.x += f.x;
    meanF.y += f.y;
    meanF.z += f.z;
  }
  meanF.x /= N;
  meanF.y /= N;
  meanF.z /= N;
  for (int i = 0; i < N; ++i) {
    real3 f = force[i];
    f.x -= meanF.x;
    f.y -= meanF.y;
    f.z -= meanF.z;
    force[i] = f;
  }

  return std::make_tuple(pos, force);
}

auto duplicatePositions(const auto &pos, const auto &force, real Lx, real Ly) {
  int N = pos.size();
  cached_vector<real3> force2(2 * N);
  cached_vector<real3> pos2(2 * N);
  for (int i = 0; i < N; ++i) {
    pos2[i] = pos[i];
    real3 p2 = pos[i];
    real3 f2 = force[i];
    p2.x += Lx;
    p2.y += Ly;
    pos2[i + N] = p2;
    force2[i] = f2;
    force2[i + N] = f2;
  }
  return std::make_tuple(pos2, force2);
}
auto computeWithDPStokes(real Lx, real Ly, auto pos, auto force) {
  const real H = 16.0; // zmax - zmin = 15 - (-1) = 16
  const real viscosity = 1.0 / (6.0 * M_PI);
  const real hydrodynamicRadius = 1.0;
  auto par =
      getDPStokesParamtersOnlyForce(Lx, Ly, H, viscosity, hydrodynamicRadius);
  par.mode = DPStokes::WallMode::slit;
  auto dpstokes = std::make_shared<DPStokes>(par);
  auto mf = dpstokes->Mdot(pos.data().get(), force.data().get(), pos.size());
  return mf;
}
} // namespace duplicate

// ----------------------------------------------------------------------------
// Verify that doubling the box in X (and repeating particles) gives
// the same mobilities (within rtol/atol) as the original cubic box.
// ----------------------------------------------------------------------------
TEST(DPStokes, PeriodicityDoubleBoxSizeXMatchesSingle) {
  using namespace duplicate;
  const real Lx = 16.0;
  const real Ly = 16.0;
  const int N = 100;
  auto [pos, force] = genPositions(N, Lx, Ly, 16.0);
  auto mf_single = computeWithDPStokes(Lx, Ly, pos, force);
  auto [pos2, force2] = duplicatePositions(pos, force, Lx, 0);
  auto mf2 = computeWithDPStokes(2 * Lx, Ly, pos2, force2);

  // compare first N entries
  for (int i = 0; i < N; ++i) {
    real3 ref = mf_single[i];
    real3 dbl = mf2[i];
    real tol = 1e-3;
    EXPECT_NEAR(dbl.x, ref.x, tol)
        << "particle " << i << " X‐component mismatch";
    EXPECT_NEAR(dbl.y, ref.y, tol)
        << "particle " << i << " Y‐component mismatch";
    EXPECT_NEAR(dbl.z, ref.z, tol)
        << "particle " << i << " Z‐component mismatch";
  }
}

// ----------------------------------------------------------------------------
// Doubling the box in Y (and repeating the particles) should reproduce
// the same mobilities as the original 16×16 slab, within rtol+atol.
// ----------------------------------------------------------------------------
TEST(DPStokes, PeriodicityDoubleBoxSizeYMatchesSingle) {
  using namespace duplicate;
  const real Lx = 16.0;
  const real Ly = 16.0;
  const int N = 100;
  auto [pos, force] = genPositions(N, Lx, Ly, 16.0);
  auto mf_single = computeWithDPStokes(Lx, Ly, pos, force);
  auto [pos2, force2] = duplicatePositions(pos, force, 0, Ly);
  auto mf2 = computeWithDPStokes(Lx, Ly * 2, pos2, force2);

  // compare first N entries
  for (int i = 0; i < N; ++i) {
    real3 ref = mf_single[i];
    real3 dbl = mf2[i];
    real tol = 1e-3;
    EXPECT_NEAR(dbl.x, ref.x, tol)
        << "particle " << i << " X‐component mismatch";
    EXPECT_NEAR(dbl.y, ref.y, tol)
        << "particle " << i << " Y‐component mismatch";
    EXPECT_NEAR(dbl.z, ref.z, tol)
        << "particle " << i << " Z‐component mismatch";
  }
}

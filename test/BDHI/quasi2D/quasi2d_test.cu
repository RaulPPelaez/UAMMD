/*Raul P. Pelaez 2023. Tests for the Quasi2D integrator

 */

#include "Integrator/Hydro/BDHI_quasi2D.cuh"
#include "gmock/gmock.h"
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <random>

using namespace uammd;
using std::make_shared;

TEST(Q2D, CanBeCreated) {
  using Scheme = BDHI::Quasi2D;
  Scheme::Parameters par;
  par.temperature = 1;
  par.viscosity = 1;
  par.dt = 0.1;
  par.hydrodynamicRadius = 1;
  par.box = Box(128);
  auto pd = std::make_shared<ParticleData>(1);
  auto bdhi = std::make_shared<Scheme>(pd, par);
}

class miniInteractor : public Interactor {
public:
  using Interactor::Interactor;

  real2 F = real2();

  void sum(Computables comp, cudaStream_t st) override {
    auto force = pd->getForce(access::location::cpu, access::mode::write);
    force[0] = make_real4(F.x, F.y, 0, 0);
    if (pg->getNumberParticles() > 1)
      force[1] = make_real4(-F.x, -F.y, 0, 0);
  }
};

template <class Scheme>
auto createBDHI(std::shared_ptr<ParticleData> pd, real temperature, real dt,
                real viscosity, real hydrodynamicRadius, real lbox) {
  typename Scheme::Parameters par;
  par.temperature = temperature;
  par.viscosity = viscosity;
  par.dt = dt;
  par.hydrodynamicRadius = hydrodynamicRadius;
  par.box = Box({lbox, lbox, 0});
  return make_shared<Scheme>(pd, par);
}

template <class Scheme>
real2 computeSelfMobility(real lbox, real hydrodynamicRadius, int dir) {
  auto pd = make_shared<ParticleData>(1);
  real viscosity = 1.12312;
  real dt = 0.1;
  auto bdhi =
      createBDHI<Scheme>(pd, 0, dt, viscosity, hydrodynamicRadius, lbox);
  auto inter = make_shared<miniInteractor>(pd, "puller");
  real F = 1.0;
  if (dir == 0)
    inter->F.x = F;
  else
    inter->F.y = F;
  bdhi->addInteractor(inter);
  int ntest = 100;
  real2 M = real2();
  // random engine
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-lbox * 0.5, lbox * 0.5);
  fori(0, ntest) {
    // random initial position
    real4 initial_pos = make_real4(dis(gen), dis(gen), 0, 0);
    pd->getPos(access::cpu, access::write)[0] = initial_pos;
    bdhi->forwardTime();
    auto new_pos = pd->getPos(access::cpu, access::read)[0];
    real2 disp = make_real2(new_pos - initial_pos);
    M += disp;
  }
  M = viscosity * M / (ntest * dt * F);
  return M;
}

TEST(Q2D, SelfMobilityQuasi2D) {
  auto pd = make_shared<ParticleData>(1);
  real a = 1.21312;
  for (real lbox = 32; lbox < 256; lbox += 32) {
    for (int dir = 0; dir < 2; dir++) {
      auto M = computeSelfMobility<BDHI::Quasi2D>(lbox * a, a, dir);
      auto Mtheo = 1.0 / (6 * M_PI * a) * (1 / (1 + 4.41 / lbox));
      if (dir == 0) {
        EXPECT_NEAR(M.x, Mtheo, 0.001)
            << "Failed for dir=" << dir << " and lbox=" << lbox << std::endl;
        EXPECT_NEAR(M.y, 0, 0.0001)
            << "Failed for dir=" << dir << " and lbox=" << lbox << std::endl;
      } else {
        EXPECT_NEAR(M.y, Mtheo, 0.001)
            << "Failed for dir=" << dir << " and lbox=" << lbox << std::endl;
        EXPECT_NEAR(M.x, 0, 0.0001)
            << "Failed for dir=" << dir << " and lbox=" << lbox << std::endl;
      }
    }
  }
}

TEST(Q2D, SelfMobilityTrue2D) {
  auto pd = make_shared<ParticleData>(1);
  real a = 1.21312;
  for (real lbox = 32; lbox < 256; lbox += 32) {
    for (int dir = 0; dir < 2; dir++) {
      auto M = computeSelfMobility<BDHI::True2D>(lbox * a, a, dir);
      auto Mtheo = 1.0 / (4 * M_PI) * (log(lbox) - 1.3105329259115095183);
      if (dir == 0) {
        EXPECT_NEAR(M.x, Mtheo, 0.001)
            << "Failed for dir=" << dir << " and lbox=" << lbox << std::endl;
        EXPECT_NEAR(M.y, 0, 0.0001)
            << "Failed for dir=" << dir << " and lbox=" << lbox << std::endl;
      } else {
        EXPECT_NEAR(M.y, Mtheo, 0.001)
            << "Failed for dir=" << dir << " and lbox=" << lbox << std::endl;
        EXPECT_NEAR(M.x, 0, 0.0001)
            << "Failed for dir=" << dir << " and lbox=" << lbox << std::endl;
      }
    }
  }
}

TEST(Q2D, ObeysFluctuationDissipationQuasi2D) {
  auto pd = make_shared<ParticleData>(1);
  real a = 1.21312;
  int navg = 50000;
  real temperature = 1.012312;
  real dt = 0.9;
  real lbox = 128 * a;
  auto bdhi = createBDHI<BDHI::Quasi2D>(pd, temperature, dt, 1, a, lbox);
  real2 avg = real2();
  // Random initial position
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-lbox * 0.5, lbox * 0.5);
  for (int i = 0; i < navg; i++) {
    auto initial_pos = make_real4(dis(gen), dis(gen), 0, 0);
    pd->getPos(access::cpu, access::write)[0] = initial_pos;
    bdhi->forwardTime();
    auto r = make_real2(pd->getPos(access::cpu, access::read)[0]) -
             make_real2(initial_pos);
    avg += r * r;
  }
  auto sigma = avg / navg;
  auto d0 = sigma / (2 * temperature * dt);
  auto Mtheo = computeSelfMobility<BDHI::Quasi2D>(lbox, a, 0).x;
  real2 err = {abs((d0.x - Mtheo) / Mtheo), abs((d0.y - Mtheo) / Mtheo)};
  EXPECT_NEAR(err.x, 0, 0.01)
      << "d0=" << d0 << " and Mtheo=" << Mtheo << std::endl;
  EXPECT_NEAR(err.y, 0, 0.01)
      << "d0=" << d0 << " and Mtheo=" << Mtheo << std::endl;
}

TEST(Q2D, ObeysFluctuationDissipationTrue2D) {
  auto pd = make_shared<ParticleData>(1);
  real a = 1.21312;
  int navg = 50000;
  real temperature = 1.012312;
  real dt = 0.9;
  real lbox = 128 * a;
  auto bdhi = createBDHI<BDHI::True2D>(pd, temperature, dt, 1, a, lbox);
  real2 avg = real2();
  // Random initial position
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-lbox * 0.5, lbox * 0.5);
  for (int i = 0; i < navg; i++) {
    auto initial_pos = make_real4(dis(gen), dis(gen), 0, 0);
    pd->getPos(access::cpu, access::write)[0] = initial_pos;
    bdhi->forwardTime();
    auto r = make_real2(pd->getPos(access::cpu, access::read)[0]) -
             make_real2(initial_pos);
    avg += r * r;
  }
  auto sigma = avg / navg;
  auto d0 = sigma / (2 * temperature * dt);
  auto Mtheo = computeSelfMobility<BDHI::True2D>(lbox, a, 0).x;
  real2 err = {abs((d0.x - Mtheo) / Mtheo), abs((d0.y - Mtheo) / Mtheo)};
  EXPECT_NEAR(err.x, 0, 0.01)
      << "d0=" << d0 << " and Mtheo=" << Mtheo << std::endl;
  EXPECT_NEAR(err.y, 0, 0.01)
      << "d0=" << d0 << " and Mtheo=" << Mtheo << std::endl;
}

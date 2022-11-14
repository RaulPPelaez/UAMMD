/*Raul P. Pelaez 2022. Tests for the VQCM DPStokes algorithm.
  Most tests should take about a second to run. A few of them will take several minutes, though.
 */
#include <fstream>
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include"Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"
#include<random>

using namespace uammd;
using DPStokesSlab_ns::DPStokes;

using DPStokesSlab_ns::cached_vector;

TEST(DPStokes, CanBeCreated){
  DPStokes::Parameters par;
  par.nx = 32;
  par.ny = 32;
  par.nz = 32;
  par.w = 6;
  par.beta = 13;
  par.alpha = par.w*0.5;
  par.mode = DPStokes::WallMode::slit;
  par.viscosity = 1;
  par.Lx = par.Ly = 1;
  par.H = 1;
  auto dpstokes = std::make_shared<DPStokes>(par);
}

//Paramters for a support of w=6
auto getDPStokesParamtersOnlyForce(real Lxy, real H, real viscosity, real hydrodynamicRadius){

  real h = hydrodynamicRadius/1.554;

  int nxy = int(Lxy/h);

  DPStokes::Parameters par;
  par.nx = nxy;
  par.ny = par.nx;
  par.nz = int(M_PI*H/(2*h));
  par.w = 6;
  par.beta = 1.714*par.w;
  par.alpha = par.w*0.5;
  par.mode = DPStokes::WallMode::slit;
  par.viscosity = viscosity;
  par.Lx = par.Ly = Lxy;
  par.H = H;
  return par;
}

//Check that a force pulling from a single particle only moves it in that direction
TEST(DPStokes, PulledParticleDoesNotMoveInOtherDirection){
  real hydrodynamicRadius = 1;
  real Lxy = 128;
  real H = Lxy;
  auto par = getDPStokesParamtersOnlyForce(Lxy, H, 1/(6*M_PI), hydrodynamicRadius);
  par.mode = DPStokes::WallMode::slit;
  auto dpstokes = std::make_shared<DPStokes>(par);
  cached_vector<real3> pos(1);
  pos[0] = {0,0,0};
  cached_vector<real3> force(1);
  force[0] = {1,0,0};
  auto mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  real3 disp =  mf[0];
  ASSERT_THAT(disp.x, ::testing::Not(::testing::DoubleNear(0.0, 1e-3)));
  ASSERT_THAT(disp.y, ::testing::DoubleNear(0.0, 1e-15));
  ASSERT_THAT(disp.z, ::testing::DoubleNear(0.0, 1e-15));
  force[0] = {0,1,0};
  mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  disp =  mf[0];
  ASSERT_THAT(disp.y, ::testing::Not(::testing::DoubleNear(0.0, 1e-3)));
  ASSERT_THAT(disp.x, ::testing::DoubleNear(0.0, 1e-15));
  ASSERT_THAT(disp.z, ::testing::DoubleNear(0.0, 1e-15));
  force[0] = {0,0,1};
  mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  disp =  mf[0];
  ASSERT_THAT(disp.z, ::testing::Not(::testing::DoubleNear(0.0, 1e-3)));
  ASSERT_THAT(disp.y, ::testing::DoubleNear(0.0, 1e-15));
  ASSERT_THAT(disp.x, ::testing::DoubleNear(0.0, 1e-15));
}

// The self mobility should converge towards the open boundary case as M0(L)\approx M0(1+a/L)
// Where "a" is an constant around 1.
// The DPStokes algorithm only guarantees about 4 digits of accuracy. So a box size of L=128 rh
//   should yield 2-3 digits around M0
//Here we measure self mobility by pulling from a single particle
TEST(DPStokes, ReproducesOpenBoundarySelfMobilityWithLargeDomain){
  real hydrodynamicRadius = 1;
  real Lxy = 128;
  real H = Lxy;
  auto par = getDPStokesParamtersOnlyForce(Lxy, H, 1/(6*M_PI), hydrodynamicRadius);
  par.mode = DPStokes::WallMode::slit;
  auto dpstokes = std::make_shared<DPStokes>(par);
  cached_vector<real3> pos(1);
  pos[0] = {0,0,0};
  cached_vector<real3> force(1);
  force[0] = {1,1,1};
  auto mf = dpstokes->Mdot(pos.data().get(), force.data().get(), 1);
  real3 disp =  mf[0];
  ASSERT_THAT(disp.x, ::testing::DoubleNear(1.0, 1e-3));
  ASSERT_THAT(disp.y, ::testing::DoubleNear(1.0, 1e-3));
  ASSERT_THAT(disp.z, ::testing::DoubleNear(1.0, 1e-3));
}

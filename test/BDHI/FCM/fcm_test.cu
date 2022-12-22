/*Raul P. Pelaez 2022. Tests for the FCM algorithm and integrator

 */
#include <fstream>
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include "Integrator/BDHI/BDHI_FCM.cuh"
#include "utils/container.h"
#include <memory>
#include<random>
#include<thrust/iterator/constant_iterator.h>
#include<thrust/host_vector.h>

using namespace uammd;

using BDHI::FCM_impl;
using Kernel = BDHI::FCM_ns::Kernels::Gaussian;
using KernelTorque = BDHI::FCM_ns::Kernels::GaussianTorque;

template<class T> using cached_vector = uninitialized_cached_vector<T>;

// int3 createGrid(real3 L, real hydrodynamicRadius, real tolerance){
//   int3 cellDim;
//   real h;
//   auto box = Box(L);
//   h = Kernel::adviseGridSize(hydrodynamicRadius, tolerance);
//   cellDim = make_int3(box.boxSize/h);
//   cellDim = nextFFTWiseSize3D(cellDim);
//   // if(par.adaptBoxSize){
//   //   box = Box(make_real3(cellDim)*h);
//   // }
//   return cellDim;
// }

auto initializeKernel(real tolerance, real3 L, int3 n){
  real h = std::min({L.x/n.x, L.y/n.y, L.z/n.z});
  auto kernel = std::make_shared<Kernel>(h, tolerance);
  return kernel;
}

auto initializeKernelTorque(real hydrodynamicRadius, real tolerance, real3 L, int3 n){
  real a = hydrodynamicRadius;
  real width = a/(pow(6*sqrt(M_PI), 1/3.));
  real h = std::min({L.x/n.x, L.y/n.y, L.z/n.z});
  auto kernelTorque = std::make_shared<KernelTorque>(width, h, tolerance);
  return kernelTorque;
}

TEST(FCM_impl, CanBeCreated){
  using FCM = FCM_impl<Kernel, KernelTorque>;
  FCM::Parameters par;
  real hydrodynamicRadius = 1;
  real3 L = {128, 128, 128};
  par.temperature = 1;
  par.viscosity = 1;
  par.tolerance = 1e-3;
  par.dt = 1;
  par.box = Box(L);
  par.cells = {128,128,128};//createGrid(L, hydrodynamicRadius, par. tolerance);
  par.kernel = initializeKernel(par.tolerance, L, par.cells);
  par.kernelTorque = initializeKernelTorque(hydrodynamicRadius, par.tolerance, L, par.cells);
  auto fcm = std::make_shared<FCM>(par);
}

real selfMobility(real hydrodynamicRadius, real viscosity, real L){
  //O(a^8) accuracy. See Hashimoto 1959.
  //With a Gaussian this expression has a minimum deviation from measuraments of 7e-7*rh at L=64*rh.
  //The translational invariance of the hydrodynamic radius however decreases arbitrarily with the tolerance.
  //Seems that this deviation decreases with L, so probably is due to the correction below missing something.
  long double rh = hydrodynamicRadius;
  long double a = rh/L;
  long double a2= a*a; long double a3 = a2*a;
  long double c = 2.83729747948061947666591710460773907l;
  long double b = 0.19457l;
  long double a6pref = 16.0l*M_PIl*M_PIl/45.0l + 630.0L*b*b;
  return  1.0l/(6.0l*M_PIl*viscosity*rh)*(1.0l-c*a+(4.0l/3.0l)*M_PIl*a3-a6pref*a3*a3);
}

//Check that the self mobility stays below tolerance at a series of random points inside the domain
// in every direction when pulling a particle
TEST(FCM_impl, SelfMobilityIsCorrectUpToTolerance){
  using FCM = FCM_impl<Kernel, KernelTorque>;
  FCM::Parameters par;
  real hydrodynamicRadius = 1.012312;
  par.viscosity = 1.12321;
  par.tolerance = 1e-7;
  par.dt = 1;
  real h = Kernel::adviseGridSize(hydrodynamicRadius, par.tolerance);
  real3 L = make_real3(128, 128, 128)*h;
  par.cells = make_int3(L/h);
  par.box = Box(L);
  //par.cells = createGrid(L, hydrodynamicRadius, par. tolerance);
  par.kernel = initializeKernel(par.tolerance, L, par.cells);
  par.kernelTorque = initializeKernelTorque(hydrodynamicRadius, par.tolerance, L, par.cells);
  par.hydrodynamicRadius = hydrodynamicRadius;
  auto fcm = std::make_shared<FCM>(par);
  int numberParticles = 1;
  cached_vector<real4> pos(numberParticles);
  real temperature = 0;
  real prefactor = 0;
  real4* torque = nullptr;
  auto force = pos;
  real3 m0;
  int ntest = 100;
  Saru rng(1234);
  for(int j = 0; j<ntest; j++){
    real3 randomPos = make_real3(rng.f(-0.5, 0.5), rng.f(-0.5, 0.5), rng.f(-0.5, 0.5))*L;
    pos[0] = make_real4(randomPos);
    for(int i = 0; i<3; i++){
      switch(i){
      case 0:
	force[0] = make_real4(1,0,0,0);
	m0 = make_real3(selfMobility(hydrodynamicRadius, par.viscosity, L.x),0,0);
	break;
      case 1:
	force[0] = make_real4(0,1,0,0);
	m0 = make_real3(0, selfMobility(hydrodynamicRadius, par.viscosity, L.x),0);
	break;
      case 2:
	force[0] = make_real4(0,0,1,0);
	m0 = make_real3(0, 0, selfMobility(hydrodynamicRadius, par.viscosity, L.x));
	break;
      }
      auto disp = fcm->computeHydrodynamicDisplacements(pos.data().get(),
							force.data().get(),
							torque,
							numberParticles, temperature, prefactor, 0);
      real3 dx = disp.first[0];
      ASSERT_THAT(dx.x, ::testing::DoubleNear(m0.x, par.tolerance));
      ASSERT_THAT(dx.y, ::testing::DoubleNear(m0.y, par.tolerance));
      ASSERT_THAT(dx.z, ::testing::DoubleNear(m0.z, par.tolerance));
    }
  }
}

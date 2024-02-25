/*Raul P. Pelaez 2022. Tests for the FCM algorithm and integrator

 */
#include <fstream>
#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include "Integrator/BDHI/BDHI_PSE.cuh"
#include "utils/container.h"
#include <memory>
#include<random>
#include<thrust/iterator/constant_iterator.h>
#include<thrust/host_vector.h>

using namespace uammd;
using BDHI::PSE;
//Different  kernels can  achieve different  maximum accuracies.   For
//instance,  the  Gaussian  kernel  can safely  achieve  8  digits  of
//accuracy in the  self mobility, but the Peskin 3pt  kernel will only
//give about 3.  Place the maximum expected accuracy  in this variable
//if you want to check a new kernel.
constexpr real expectedAccuracy = 1e-8;
//Although the  Gaussian kernel  should be able  to achieve  even more
//accuracy, beyond 8 digits the  approximate solution for the periodic
//correction of the self mobility I am using starts to fail.

template<class T> using cached_vector = uninitialized_cached_vector<T>;

TEST(PSE, CanBeCreated){
  PSE::Parameters par;
  real hydrodynamicRadius = 1;
  real3 L = {128, 128, 128};
  par.temperature = 1;
  par.viscosity = 1;
  par.tolerance = 1e-3;
  par.dt = 1;
  par.box = Box(L);
  par.psi = 1.0;
  auto pd = std::make_shared<ParticleData>(1);
  auto pse = std::make_shared<PSE>(pd, par);
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
TEST(PSE, SelfMobilityIsCorrectUpToTolerance){
  PSE::Parameters par;
  real hydrodynamicRadius = 1.012312;
  par.viscosity = 1.12321;
  par.tolerance = expectedAccuracy;
  par.dt = 1;
  real3 L = make_real3(128*hydrodynamicRadius);
  par.box = Box(L);
  par.hydrodynamicRadius = hydrodynamicRadius;
  par.psi = 1.0;
  int numberParticles = 1;
  auto pd = std::make_shared<ParticleData>(numberParticles);
  auto pse = std::make_shared<PSE>(pd, par);
  cached_vector<real4> force(numberParticles);
  cached_vector<real3> MF(numberParticles);
  real temperature = 0;
  real prefactor = 0;
  real3 m0;
  int ntest = 20;
  Saru rng(1234);
  for(int j = 0; j<ntest; j++){
    {
      auto pos = pd->getPos(access::cpu, access::write);
      real3 randomPos = make_real3(rng.f(-0.5, 0.5), rng.f(-0.5, 0.5), rng.f(-0.5, 0.5))*L;
      pos[0] = make_real4(randomPos);
    }
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
      pse->computeHydrodynamicDisplacements(force.data().get(),
					    MF.data().get(), temperature, prefactor, 0);
      real3 dx = MF[0];
      ASSERT_THAT(dx.x, ::testing::DoubleNear(m0.x, par.tolerance));
      ASSERT_THAT(dx.y, ::testing::DoubleNear(m0.y, par.tolerance));
      ASSERT_THAT(dx.z, ::testing::DoubleNear(m0.z, par.tolerance));
    }
  }
}

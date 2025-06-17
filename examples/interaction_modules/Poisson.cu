/*Raul P. Pelaez 2021. Triply periodic electrostatics.
  The Poisson Interactor in SpectralEwaldPoisson.cuh encodes two solvers for the
Poisson equation in a triply periodic domain.

  One is fully spectral and the other one has Ewald splitting. Go to the
relevant wiki page if you want more information about the algorithms [1].

  You can use this interaction module to compute electrostatic interactions
between particles.

  In this code you will find a copy pastable function to create a Poisson module
that you can then add to an Integrator.


[1] https://github.com/RaulPPelaez/UAMMD/wiki/SpectralEwaldPoisson
 */

#include "Interactor/SpectralEwaldPoisson.cuh"
#include "uammd.cuh"
using namespace uammd;

// This struct contains the basic uammd modules for convenience.
struct UAMMD {
  std::shared_ptr<System> sys;
  std::shared_ptr<ParticleData> pd;
};

// Creates an instance of a Poisson Interactor that can be added to an
// integrator. When using Electrostatics remember to initialize particle charges
// via ParticleData::getCharges in addition to positions.
std::shared_ptr<Interactor> createElectrostaticInteractor(UAMMD sim) {
  Poisson::Parameters par;
  par.box = Box(make_real3(32, 32, 32));
  // Permittivity
  par.epsilon = 1;
  // Poisson models charges as Gaussian sources, this is their width
  par.gw = 1.0;
  // Desired overall error tolerance of the algorithm.
  par.tolerance = 1e-8;
  // If gw << Lbox the default algorithm can become inefficient.
  // In order to overcome this limitation an Ewald splitted implementation is
  // also provided and will be enabled if
  //  the split parameter is set.
  // Split controls how the load is balanced between the two sections of the
  // algorithm A low split will be beneficial to a low density system and the
  // other way around. An optimal split always exists and it might be a good
  // idea to look for it in a case by case basis. par.split = 1.0;
  auto poisson = std::make_shared<Poisson>(sim.pd, par);
  return poisson;
}

int main(int argc, char *argv[]) { return 0; }

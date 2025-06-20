/*Raul P. Pelaez 2017-2020. Positively Split Edwald Rotne-Prager-Yamakawa
Brownian Dynamics with Hydrodynamic interactions.

Initialization

*/
#include "Integrator/BDHI/BDHI_PSE.cuh"
namespace uammd {
namespace BDHI {
namespace pse_ns {
void checkInputValidity(BDHI::PSE::Parameters par) {
  real3 L = par.box.boxSize;
  if (L.x == real(0.0) && L.y == real(0.0) && L.z == real(0.0)) {
    System::log<System::EXCEPTION>(
        "[BDHI::PSE] Box of size zero detected, cannot work without a box! "
        "(make sure a box parameter was passed)");
    throw std::invalid_argument("Box of size zero detected");
  }
  if (L.x != L.y || L.y != L.z || L.x != L.z) {
    System::log<System::WARNING>(
        "[BDHI::PSE] Non cubic boxes are not really tested!");
  }
  if (par.tolerance > 0.1) {
    System::log<System::EXCEPTION>(
        "[BDHI::PSE] Tolerance too high, this will lead to numerical "
        "instability. Tolerance should be less than 0.1 (10% error).");
    throw std::invalid_argument("Tolerance too high");
  }
}

long double computeSelfMobility(PSE::Parameters par) {
  // O(a^8) accuracy. See Hashimoto 1959.
  // With a Gaussian this expression has a minimum deviation from measuraments
  // of 7e-7*rh at L=64*rh. The translational invariance of the hydrodynamic
  // radius however decreases arbitrarily with the tolerance. Seems that this
  // deviation decreases with L, so probably is due to the correction below
  // missing something.
  long double rh = par.hydrodynamicRadius;
  long double L = par.box.boxSize.x;
  long double a = rh / L;
  long double a2 = a * a;
  long double a3 = a2 * a;
  long double c = 2.83729747948061947666591710460773907l;
  long double b = 0.19457l;
  long double a6pref = 16.0l * M_PIl * M_PIl / 45.0l + 630.0L * b * b;
  return 1.0l / (6.0l * M_PIl * par.viscosity * rh) *
         (1.0l - c * a + (4.0l / 3.0l) * M_PIl * a3 - a6pref * a3 * a3);
}

} // namespace pse_ns

PSE::PSE(shared_ptr<ParticleGroup> pg, Parameters par)
    : pg(pg), hydrodynamicRadius(par.hydrodynamicRadius),
      temperature(par.temperature), dt(par.dt) {
  System::log<System::MESSAGE>("[BDHI::PSE] Initialized");
  this->M0 = pse_ns::computeSelfMobility(par);
  System::log<System::MESSAGE>("[BDHI::PSE] Self mobility: %f", M0);
  pse_ns::checkInputValidity(par);
  nearField = std::make_shared<pse_ns::NearField>(par, pg);
  farField = std::make_shared<pse_ns::FarField>(
      pg->getParticleData()->getSystem(), par);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();
}

} // namespace BDHI
} // namespace uammd

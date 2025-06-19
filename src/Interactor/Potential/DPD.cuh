/* Raul P. Pelaez 2018-2021. Dissipative Particle Dynamics potential.

   A DPD simulation can be seen as a regular Molecular Dynamics simulation with
a special interaction force between particles [1]. This file implements a
Potential entity that when used with a VerletNVE integrator (through a
PairForces interactor) will produce a DPD simulation.

References:
[1] On the numerical treatment of dissipative particle dynamics and related
systems. Leimkuhler and Shang 2015. https://doi.org/10.1016/j.jcp.2014.09.008

 */

#include "Interactor/Interactor.cuh"
#include "ParticleData/ParticleData.cuh"
#include "System/System.h"
#include "misc/ParameterUpdatable.h"
#include "third_party/saruprng.cuh"
#include "third_party/type_names.h"
#include "utils/Box.cuh"

namespace uammd {
namespace Potential {
struct DefaultDissipation : public ParameterUpdatable {
  real A;
  real gamma;
  real sigma; // Random force strength
  real temperature, dt;
  DefaultDissipation(real A, real gamma, real temperature, real dt)
      : A(A), gamma(gamma), temperature(temperature), dt(dt) {
    this->sigma = sqrt(2.0 * temperature / dt);
    System::log<System::MESSAGE>("[Potential::DPD] Created with A: %f, gamma: "
                                 "%f, temperature: %f, dt: %f",
                                 A, gamma, temperature, dt);
  }

  __device__ auto operator()(int i, int j, real3 rij, real3 vij, real cutoff,
                             Saru &rng) const {
    const real rmod = sqrt(dot(rij, rij));
    const real invrmod = real(1.0) / rmod;
    const auto g = gamma;
    // This weight function is arbitrary as long as wd = wr*wr
    const real wr = real(1.0) - rmod / cutoff;
    const real Fc = A * wr * invrmod;
    const real wd = wr * wr; // Wd must be such as wd = wr^2 to ensure
                             // fluctuation dissipation balance
    const auto Fd = -g * wd * invrmod * invrmod * dot(rij, vij);
    const real Fr = rng.gf(real(0.0), sigma * sqrt(g) * wr * invrmod).x;
    return (Fc + Fd + Fr) * rij;
  }

  virtual void updateTemperature(real newTemp) override {
    temperature = newTemp;
    sigma = sqrt(2.0 * temperature) / sqrt(dt);
  }

  virtual void updateTimeStep(real newdt) override {
    dt = newdt;
    sigma = sqrt(2.0 * temperature) / sqrt(dt);
  }
};

struct TransversalDissipation : public ParameterUpdatable {
  real g_par, g_perp;
  real sigma; // Random force strength, must be such as sigma =
              // sqrt(2*kT*gamma)/sqrt(dt)
  real temperature, dt;
  real A;
  TransversalDissipation(real g_par, real g_perp, real A, real temperature,
                         real dt)
      : g_par(g_par), g_perp(g_perp), temperature(temperature), dt(dt), A(A) {
    this->sigma = sqrt(2.0 * temperature) / sqrt(dt);
    if (g_par < (4.0 / 3.0) * g_perp) {
      throw std::runtime_error("[TransversalDissipation] g_par must be greater "
                               "than 4/3 * g_perp, found g_par: " +
                               std::to_string(g_par) +
                               " g_perp: " + std::to_string(g_perp));
    }
    if (g_perp < 0) {
      throw std::runtime_error(
          "[TransversalDissipation] g_perp must be non-negative");
    }
  }

  __device__ __host__ auto operator()(int i, int j, real3 rij, real3 vij,
                                      real cutoff, Saru &rng) const {
    const real rmod = sqrt(dot(rij, rij));
    const real wr = real(1.0) - rmod / cutoff;
    const auto eij = rij / rmod;
    const auto Fc = A * wr * eij;
    const auto g_par_r = g_par * wr * wr;
    const auto g_perp_r = g_perp * wr * wr;
    const auto Fd = this->dissipative(eij, vij, g_par_r, g_perp_r);
    const auto Fr = this->fluctuation(eij, rng, g_par_r, g_perp_r);
    return Fc + Fd + Fr;
  }

  __device__ __host__ real3 dissipative(real3 eij, real3 vij, real g_par,
                                        real g_perp) const {
    // (eij\dyadic eij )\dot vij
    const auto vdyadic = make_real3(
        vij.x * eij.x * eij.x + vij.y * eij.x * eij.y + vij.z * eij.x * eij.z,
        vij.x * eij.y * eij.x + vij.y * eij.y * eij.y + vij.z * eij.y * eij.z,
        vij.x * eij.z * eij.x + vij.y * eij.z * eij.y + vij.z * eij.z * eij.z);
    const auto videntity = vij;
    const auto gv = g_perp * videntity + (g_par - g_perp) * vdyadic;
    return -gv;
  }

  __device__ __host__ real3 fluctuation(real3 eij, Saru &rng, real g_par,
                                        real g_perp) const {
    const auto A = g_perp;
    const auto B = g_par - g_perp;
    const auto Atil = sqrt(2 * A);
    const auto Btil = sqrt(3 * B - A);
    const auto noiseX = make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
    const auto noiseY = make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
    const auto noiseZ = make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
    const auto trNoise = real(1.0 / 3.0) * (noiseX.x + noiseY.y + noiseZ.z);
    const auto noiseA =
        real(0.5) *
            (make_real3(
                (noiseX.x + noiseX.x) * eij.x + (noiseX.y + noiseY.x) * eij.y +
                    (noiseX.z + noiseZ.x) * eij.z,
                (noiseY.x + noiseX.y) * eij.x + (noiseY.y + noiseY.y) * eij.y +
                    (noiseY.z + noiseZ.y) * eij.z,
                (noiseZ.x + noiseX.z) * eij.x + (noiseZ.y + noiseY.z) * eij.y +
                    (noiseZ.z + noiseZ.z) * eij.z)) -
        trNoise * eij;
    const auto noiseB = trNoise * eij;
    const auto fluc = sigma * (Atil * noiseA + Btil * noiseB);
    return fluc;
  }
  virtual void updateTemperature(real newTemp) override {
    temperature = newTemp;
    sigma = sqrt(2.0 * temperature) / sqrt(dt);
  }

  virtual void updateTimeStep(real newdt) override {
    dt = newdt;
    sigma = sqrt(2.0 * temperature) / sqrt(dt);
  }
};

template <class DissipativeStrength = DefaultDissipation>
class DPD_impl : public ParameterUpdatable {
protected:
  uint step;
  real rcut;
  std::shared_ptr<DissipativeStrength> gamma; // Dissipative force strength
  real dt;

public:
  struct Parameters {
    real cutOff = 1;
    real dt = 0;
    std::shared_ptr<DissipativeStrength> gamma;
  };

  // The system parameter is deprecated and actually unused, the
  //  other constructor is left here for retrocompatibility
  DPD_impl(Parameters par) : DPD_impl(nullptr, par) {}

  DPD_impl(shared_ptr<System> sys, Parameters par)
      : rcut(par.cutOff), dt(par.dt), gamma(par.gamma) {
    System::log<System::MESSAGE>("[Potential::DPD] Created");
    step = 0;
    System::log<System::MESSAGE>("[Potential::DPD] Cut off: %f", rcut);
    printGamma();
  }
  void printGamma();

  ~DPD_impl() { System::log<System::MESSAGE>("[Potential::DPD] Destroyed"); }

  real getCutOff() { return rcut; }

  virtual void updateTemperature(real newTemp) override {
    gamma->updateTemperature(newTemp);
  }

  virtual void updateTimeStep(real newdt) override {
    dt = newdt;
    gamma->updateTimeStep(newdt);
  }

  struct ForceTransverser {
    real4 *pos;
    real3 *vel;
    real4 *force;
    Box box;
    uint seed; // A random seed
    uint step; // Current time step
    uint N;
    real cutoff;
    DissipativeStrength gamma;

    using returnInfo = real3;

    struct Info {
      real3 vel;
      int id;
    };

    inline __device__ returnInfo compute(const real4 &pi, const real4 &pj,
                                         const Info &infoi, const Info &infoj) {
      real3 rij = box.apply_pbc(make_real3(pi) - make_real3(pj));
      real3 vij = make_real3(infoi.vel) - make_real3(infoj.vel);
      // The random force must be such as Frij = Frji, we achieve this by
      // seeding the RNG the same for pairs ij and ji
      uint i = infoi.id;
      uint j = infoj.id;
      if (i > j)
        thrust::swap(i, j);
      const uint ij = i + N * j;
      Saru rng(ij, seed, step);
      const real r2 = dot(rij, rij);
      // There is an indetermination at r=0
      // The force is 0 beyond rcut
      if (r2 == real(0) || r2 >= cutoff * cutoff)
        return make_real3(0);
      return gamma(i, j, rij, vij, cutoff, rng);
    }

    inline __device__ Info getInfo(int pi) { return {vel[pi], pi}; }

    inline __device__ void set(uint pi, const returnInfo &total) {
      force[pi] += make_real4(total);
    }
  };

  ForceTransverser getTransverser(Interactor::Computables comp, Box box,
                                  shared_ptr<ParticleData> pd) {
    if (comp.energy || comp.stress || comp.virial) {
      System::log<System::EXCEPTION>(
          "[DPD] This Potential can only compute forces");
      throw std::runtime_error("DPD: This Potential can only compute forces");
    }
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto vel = pd->getVel(access::location::gpu, access::mode::read);
    auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
    static uint seed = pd->getSystem()->rng().next32();
    step++;
    const uint N = pd->getNumParticles();
    return ForceTransverser{pos.raw(), vel.raw(), force.raw(), box,   seed,
                            step,      N,         rcut,        *gamma};
  }
};

template <> inline void DPD_impl<DefaultDissipation>::printGamma() {
  System::log<System::MESSAGE>("[Potential::DPD] gamma: %f", gamma->gamma);
}

template <class T> inline void DPD_impl<T>::printGamma() {
  System::log<System::MESSAGE>("[Potential::DPD] Using %s for dissipation",
                               type_name<T>().c_str());
}

using DPD = DPD_impl<>;
} // namespace Potential

} // namespace uammd

/* Raul P. Pelaez 2025. Connection between the DPD dissipation kernels and the
   PairForces module

 */
#pragma once
#include "DissipationKernels.cuh"
namespace uammd {
namespace dpd {

template <DissipationKernel Kernel>
class DPDPotential : public ParameterUpdatable {
private:
  uint step;
  real rcut;
  std::shared_ptr<Kernel> dissipation; // Dissipative force strength

public:
  struct Parameters {
    real cutOff = 1;
    std::shared_ptr<Kernel> dissipation;
  };

  DPDPotential(Parameters par) : DPDPotential(nullptr, par) {}

  DPDPotential(shared_ptr<System> sys, Parameters par)
      : rcut(par.cutOff), dissipation(par.dissipation) {
    System::log<System::MESSAGE>("[Potential::DPD] Created");
    step = 0;
    System::log<System::MESSAGE>("[Potential::DPD] Cut off: %f", rcut);
  }

  ~DPDPotential() {
    System::log<System::MESSAGE>("[Potential::DPD] Destroyed");
  }

  real getCutOff() { return rcut; }

  virtual void updateTemperature(real newTemp) override {
    dissipation->updateTemperature(newTemp);
  }

  virtual void updateTimeStep(real newdt) override {
    dissipation->updateTimeStep(newdt);
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
    Kernel dissipation;

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
      return dissipation(i, j, rij, vij, cutoff, rng);
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
    auto pos = pd->getPos(access::gpu, access::read);
    auto vel = pd->getVel(access::gpu, access::read);
    auto force = pd->getForce(access::gpu, access::readwrite);
    static uint seed = pd->getSystem()->rng().next32();
    step++;
    const uint N = pd->getNumParticles();
    return ForceTransverser{pos.raw(), vel.raw(), force.raw(), box, seed, step,
                            N,         rcut,      *dissipation};
  }
};
} // namespace dpd
} // namespace uammd

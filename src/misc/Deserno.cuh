/*Raul P. Pelaez 2019. The Deserno model

This file contains an Interactor that given files with bonds describing Deserno
lipids can compute the forces and energies given by the Deserno potential[1].

USAGE:


//Particles with type 0 are considered heads and type 1 is considered tails.
Deserno::Parameters params;
params.box = box;
params.fileHarmonic = harmonicbonds;
params.fileFENE = FENEbonds;
params.wc = 1.2;   //The tune parameter in [1]
params.epsilon = 1.0; //WCA epsilon
params.sigma = 1.0;  //WCA sigma
auto deserno = make_shared<Deserno>(pd, sys, params);

myIntegrator->addInteractor(deserno);


REFERENCES:

[1] Tunable generic model for fluid bilayer membranes. Ira R. Cooke, Kurt
Kremer, and Markus Deserno DOI: 10.1103/PhysRevE.72.011506
 */
#ifndef DESERNO_CUH
#define DESERNO_CUH
#include "Interactor/BondedForces.cuh"
#include "Interactor/NeighbourList/CellList.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/Potential/Potential.cuh"

namespace uammd {

namespace DesernoPotential {

struct NonBonded_Functor {
  struct InputPairParameters {
    real cutOff, sigma, epsilon, wc, rc;
  };

  struct __align__(16) PairParameters {
    real rc, wc;
    real b2, epsilon;
  };
  // Return F/r
  static inline __host__ __device__ real force(const real &r2,
                                               const PairParameters &params) {
    const real rcpluswc2 = (params.rc + params.wc) * (params.rc + params.wc);
    if (r2 > rcpluswc2)
      return 0;
    real fmod = real(0);
    if (r2 < params.rc * params.rc) {
      const real invr2 = params.b2 / r2;
      const real invr6 = invr2 * invr2 * invr2;
      const real invr8 = invr6 * invr2;
      fmod += params.epsilon * (real(-48.0) * invr6 + real(24.0)) * invr8 /
              params.b2;
    } else if (r2 < rcpluswc2) {
      real r = sqrt(r2);
      real s, c;
      sincospi((r - params.rc) / (real(2.0) * params.wc), &s, &c);
      fmod = params.epsilon * real(M_PI) * s * c / (params.wc * r);
    }

    return fmod;
  }

  static inline __host__ __device__ real energy(const real &r2,
                                                const PairParameters &params) {
    const real rcpluswc2 = (params.rc + params.wc) * (params.rc + params.wc);
    if (r2 > rcpluswc2)
      return 0;
    real E = 0;
    if (r2 < params.rc * params.rc) {
      const real invr2 = params.b2 / r2;
      const real invr6 = invr2 * invr2 * invr2;
      E += params.epsilon * real(4.0) * invr6 * (invr6 - real(1.0)) +
           params.epsilon * (params.wc > 0 ? real(1.0) : real(0.0));
    } else if (r2 < rcpluswc2) {
      const real c = cospi((sqrt(r2) - params.rc) / (real(2.0) * params.wc));
      E -= params.epsilon * c * c;
    }

    return E;
  }
  static inline __host__ PairParameters
  processPairParameters(InputPairParameters in_par) {
    PairParameters params;
    params.rc = in_par.rc;
    params.b2 = in_par.sigma * in_par.sigma;
    params.epsilon = in_par.epsilon;
    params.wc = in_par.wc;
    return params;
  }
};
using NonBonded = Potential::Radial<NonBonded_Functor>;

} // namespace DesernoPotential

class Deserno : public Interactor {
  using PairForces = PairForces<DesernoPotential::NonBonded, CellList>;
  shared_ptr<PairForces> nonBonded;
  using BondedHarmonic = BondedForces<BondedType::HarmonicPBC>;
  shared_ptr<BondedHarmonic> bondedHarmonic;
  using BondedFENE = BondedForces<BondedType::FENEPBC>;
  shared_ptr<BondedFENE> bondedFENE;

  cudaStream_t stream;

public:
  struct Parameters {
    Box box;
    real wc;
    std::string fileHarmonic, fileFENE;
    real epsilon = 1;
    real sigma = 1;
  };
  Deserno(shared_ptr<ParticleData> pd, shared_ptr<ParticleGroup> pg,
          shared_ptr<System> sys, Parameters par);

  // If no group is provided, a group with all particles is assumed
  Deserno(shared_ptr<ParticleData> pd, shared_ptr<System> sys, Parameters par)
      : Deserno(pd, std::make_shared<ParticleGroup>(pd, sys), sys, par) {}

  ~Deserno() = default;

  void sumForce(cudaStream_t st) override;
  real sumEnergy() override;
};

} // namespace uammd
#include "Deserno.cu"
#endif
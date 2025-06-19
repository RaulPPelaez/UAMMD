/*Raul P. Pelaez 2019. The Deserno potential.
See Deserno.cuh
*/
#include "Deserno.cuh"
#include <fstream>
namespace uammd {

Deserno::Deserno(shared_ptr<ParticleData> pd, shared_ptr<ParticleGroup> pg,
                 shared_ptr<System> sys, Parameters par)
    : Interactor(pd, pg, sys, "Deserno") {

  auto pot = std::make_shared<DesernoPotential::NonBonded>(sys);
  {
    DesernoPotential::NonBonded::InputPairParameters param;
    param.epsilon = par.epsilon;
    // Tails
    param.sigma = par.sigma;
    real wc = par.wc * param.sigma;
    real rc = pow(2, 1 / 6.0) * param.sigma;
    param.cutOff = rc + wc;
    param.rc = rc;
    param.wc = wc;
    pot->setPotParameters(1, 1, param);

    // Heads and head+tail
    param.sigma = 0.95 * par.sigma;
    wc = par.wc * param.sigma;
    rc = pow(2, 1 / 6.0) * param.sigma;
    param.cutOff = rc;
    param.rc = rc;
    param.wc = 0;
    pot->setPotParameters(0, 0, param);
    pot->setPotParameters(0, 1, param);
  }

  {
    PairForces::Parameters params;
    params.box = par.box;
    nonBonded = std::make_shared<PairForces>(pd, pg, sys, params, pot);
  }
  {
    BondedHarmonic::Parameters params;
    params.file = par.fileHarmonic;
    bondedHarmonic = std::make_shared<BondedHarmonic>(
        pd, sys, params, BondedType::HarmonicPBC(par.box));
  }
  {
    BondedFENE::Parameters params;
    params.file = par.fileFENE;
    bondedFENE = std::make_shared<BondedFENE>(pd, sys, params,
                                              BondedType::FENEPBC(par.box));
  }
}

void Deserno::sumForce(cudaStream_t st) {
  nonBonded->sumForce(st);
  bondedHarmonic->sumForce(st);
  bondedFENE->sumForce(st);
}
real Deserno::sumEnergy() {
  return nonBonded->sumEnergy() + bondedHarmonic->sumEnergy() +
         bondedFENE->sumEnergy();
}

} // namespace uammd
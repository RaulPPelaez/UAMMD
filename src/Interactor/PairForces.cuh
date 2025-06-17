/*Raul P. Pelaez 2017-2021. PairForces definition.

  PairForces Module is an interactor that computes forces, energies and/or
  virials between pairs of particle closer to a given cut off distance. If the
  cut off reaches a certain threshold the algorithm switches to n-body.

  A Potential describing the interaction must be provided.
  See misc/Potential.cuh and https://github.com/RaulPPelaez/UAMMD/wiki/Potential
  for more info on potentials and how to implement them.

  See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces for more info.
*/

#ifndef PAIRFORCES_H
#define PAIRFORCES_H

#include "Interactor/Interactor.cuh"
#include "Interactor/NBody.cuh"
#include "Interactor/NeighbourList/CellList.cuh"
#include "third_party/type_names.h"

namespace uammd {
template <class Potential, class NeighbourList = CellList>
class PairForces : public Interactor,
                   public ParameterUpdatableDelegate<Potential> {
public:
  struct Parameters {
    Box box = Box(std::numeric_limits<real>::infinity());
    shared_ptr<NeighbourList> nl = shared_ptr<NeighbourList>(nullptr);
  };
  PairForces(shared_ptr<ParticleData> pd, Parameters par = Parameters(),
             shared_ptr<Potential> pot = std::make_shared<Potential>())
      : PairForces(std::make_shared<ParticleGroup>(pd, "All"), par, pot) {}

  PairForces(shared_ptr<ParticleGroup> pg, Parameters par = Parameters(),
             shared_ptr<Potential> pot = std::make_shared<Potential>());

  ~PairForces() { sys->log<System::DEBUG>("[PairForces] Destroyed."); }

  void updateBox(Box box) override {
    sys->log<System::DEBUG3>("[PairForces] Box updated.");
    this->box = box;
    ParameterUpdatableDelegate<Potential>::updateBox(box);
  }

  void sum(Computables comp, cudaStream_t st) override;

  template <class Transverser>
  void sumTransverser(Transverser &tr, cudaStream_t st);

  void print_info() {
    sys->log<System::MESSAGE>("[PairForces] Using: %s Neighbour List.",
                              type_name<NeighbourList>());
    sys->log<System::MESSAGE>("[PairForces] Using: %s potential.",
                              type_name<Potential>());
  }

private:
  shared_ptr<NeighbourList> nl;
  shared_ptr<NBody> nb;
  shared_ptr<Potential> pot;
  Box box;
  real rcut;
};
} // namespace uammd

#include "PairForces.cu"

#endif

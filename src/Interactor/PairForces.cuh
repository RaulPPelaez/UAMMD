/*Raul P. Pelaez 2017-2021. PairForces definition.

  PairForces Module is an interactor that computes forces and/or energies between pairs of particle closer to a given cut off distance.
  If the cut off reaches a certain threshold the algorithm switches to n-body.

  A Potential describing the interaction must be provided. 
  See misc/Potential.cuh and https://github.com/RaulPPelaez/UAMMD/wiki/Potential for more info on potentials and how to implement them.

  See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces for more info.
*/

#ifndef PAIRFORCES_H
#define PAIRFORCES_H

#include"Interactor/Interactor.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/NBody.cuh"
#include"third_party/type_names.h"

namespace uammd{
  template<class Potential, class NeighbourList = CellList>
  class PairForces: public Interactor, public ParameterUpdatableDelegate<Potential>{
  public:
    struct Parameters{
      Box box;
      shared_ptr<NeighbourList> nl = shared_ptr<NeighbourList>(nullptr);
    };
    PairForces(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       Parameters par,
	       shared_ptr<Potential> pot = std::make_shared<Potential>());

    PairForces(shared_ptr<ParticleData> pd, shared_ptr<System> sys,
               Parameters par,
               shared_ptr<Potential> pot = std::make_shared<Potential>())
        : PairForces(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys,
                     par, pot) {
      }

    ~PairForces(){
      sys->log<System::DEBUG>("[PairForces] Destroyed.");
    }

    void updateBox(Box box) override{
      sys->log<System::DEBUG3>("[PairForces] Box updated.");
      this->box = box;
      ParameterUpdatableDelegate<Potential>::updateBox(box);
    }

    void sumForce(cudaStream_t st) override;

    real sumEnergy() override;

    real sumForceEnergy(cudaStream_t st) override;

    template<class Transverser>
    void sumTransverser(Transverser &tr, cudaStream_t st);

    void print_info(){
      sys->log<System::MESSAGE>("[PairForces] Using: %s Neighbour List.", type_name<NeighbourList>());
      sys->log<System::MESSAGE>("[PairForces] Using: %s potential.", type_name<Potential>());
    }


  private:
    shared_ptr<NeighbourList> nl;
    shared_ptr<NBody> nb;
    shared_ptr<Potential> pot;
    Box box;
    real rcut;
  };
}

#include"PairForces.cu"

#endif


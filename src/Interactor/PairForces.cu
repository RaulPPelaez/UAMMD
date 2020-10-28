/*Raul P. Pelaez 2017. PairForces definition.

  PairForces Module is an interactor that computes short range forces.

  Computes the interaction between neighbour particles (pairs of particles closer tan rcut).
  If the value of rcut reaches a certain threshold, the computation will be

  For that, it uses a NeighbourList or an Nbody interaction and computes the force given by Potential for each pair of particles. It sums the force for all neighbours of every particle.

  See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces   for more info.

  See misc/Potential.cuh and https://github.com/RaulPPelaez/UAMMD/wiki/Potential for more info on potentials and how to implement them.
*/

#include<cub/cub.cuh>
#include"PairForces.cuh"
#include"utils/GPUUtils.cuh"
#include"utils/cxx_utils.h"
#include"Potential/PotentialUtils.cuh"
#include"utils/TransverserUtils.cuh"
namespace uammd{
  template<class MyPotential, class NL>
  PairForces<MyPotential, NL>::PairForces(shared_ptr<ParticleData> pd,
					shared_ptr<ParticleGroup> pg,
					shared_ptr<System> sys,
					Parameters par,
					shared_ptr<MyPotential> pot):
    Interactor(pd, pg, sys,
	       "PairForces/" +
	       stringUtils::removePattern(type_name<NL>(), "uammd::") +
	       "/" +
	       stringUtils::removePattern(type_name<MyPotential>(), "uammd::")),
    box(par.box),
    pot(pot),
    nl(par.nl),
    nb(nullptr)
  {
    this->setDelegate(pot.get());
  }


  template<class MyPotential, class NL>
  template<class Transverser>
  void PairForces<MyPotential, NL>::sumTransverser(Transverser &tr, cudaStream_t st){
    this->rcut = pot->getCutOff();
    sys->log<System::DEBUG3>("[PairForces] Using cutOff: %f", this->rcut);
    bool useNeighbourList = true;
    int3 ncells = make_int3(box.boxSize/rcut);
    if(ncells.x <=3 and ncells.y <= 3 and ncells.z <=3){
      useNeighbourList = false;
    }
    if(useNeighbourList){
      if(!nl){
	nl = std::make_shared<NL>(pd, pg, sys);
      }
      nl->update(box, rcut, st);
      sys->log<System::DEBUG2>("[PairForces] Transversing neighbour list");
      nl->transverseList(tr, st);
    }
    else{
      if(!nb){
	nb = std::make_shared<NBody>(pd, pg, sys);
      }
      sys->log<System::DEBUG2>("[PairForces] Transversing NBody");
      nb->transverse(tr, st);
    }
  }

  template<class MyPotential, class NL>
  void PairForces<MyPotential, NL>::sumForce(cudaStream_t st){
    sys->log<System::DEBUG1>("[PairForces] Summing forces");
    auto ft = pot->getForceTransverser(box, pd);
    this->sumTransverser(ft, st);
  }

  template<class MyPotential, class NL>
  real PairForces<MyPotential, NL>::sumEnergy(){
    sys->log<System::DEBUG1>("[PairForces] Summing Energy");
    cudaStream_t st = 0;
    auto et = Potential::getIfHasEnergyTransverser<MyPotential>::get(pot, box, pd);
    constexpr bool isnull = std::is_same<decltype(et), BasicNullTransverser>::value;
    if(isnull) return 0.0;
    else
      this->sumTransverser(et, st);
    return 0;
  }

  template<class MyPotential, class NL>
  real PairForces<MyPotential, NL>::sumForceEnergy(cudaStream_t st){
    sys->log<System::DEBUG1>("[PairForces] Summing Force and Energy");
    auto ft = pot->getForceEnergyTransverser(box, pd);
    this->sumTransverser(ft, st);
    return 0;
  }


}

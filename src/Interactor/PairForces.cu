/*Raul P. Pelaez 2017. PairForces definition.

  PairForces Module is an interactor that computes short range forces.

  Computes the interaction between neighbour particles (pairs of particles closer tan rcut).
  If the value of rcut reaches a certain threshold, the computation will be

  For that, it uses a NeighbourList or an Nbody interaction and computes the force given by Potential for each pair of particles. It sums the force for all neighbours of every particle.

  See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces   for more info.

  See misc/Potential.cuh and https://github.com/RaulPPelaez/UAMMD/wiki/Potential for more info on potentials and how to implement them.
*/

#include<third_party/cub/cub.cuh>

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


    //If the cutoff distance is too high, fall back to an NBody interaction
    int3 ncells = make_int3(box.boxSize/rcut);

    if(ncells.x <=3 || ncells.y <= 3 || (ncells.z <=3 && ncells.z>0) ){
      useNeighbourList = false;
    }



    if(useNeighbourList){
      if(!nl){
	//A neighbour list must know just my system information at construction
	nl = std::make_shared<NL>(pd, pg, sys);
      }

      //Update neighbour list. It is smart enough to do it only when it is necessary,
      // so do not fear calling it several times.
      nl->updateNeighbourList(box, rcut, st);

      sys->log<System::DEBUG2>("[PairForces] Transversing neighbour list");

      //nl->transverseListWithNeighbourList(tr, st);
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
    //This will get the EnergyTransverser of the potential if it is defined, and a null transverser otherwise
    auto et = Potential::getIfHasEnergyTransverser<MyPotential>::get(pot, box, pd);
    //If a null transverser has been issued, just return 0
    constexpr bool isnull = std::is_same<decltype(et), BasicNullTransverser>::value;
    if(isnull) return 0.0;
    else
      this->sumTransverser(et, st);
    return 0;
  }


}

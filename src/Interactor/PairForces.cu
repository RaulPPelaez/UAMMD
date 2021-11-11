/*Raul P. Pelaez 2017-2021. PairForces definition.

  PairForces Module is an interactor that computes forces and/or energies between pairs of particle closer to a given cut off distance.
  If the cut off reaches a certain threshold the algorithm switches to n-body.

  A Potential describing the interaction must be provided. 
  See misc/Potential.cuh and https://github.com/RaulPPelaez/UAMMD/wiki/Potential for more info on potentials and how to implement them.

  See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces for more info.
*/

#include"PairForces.cuh"
#include"Potential/PotentialUtils.cuh"
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
    constexpr bool hasForceTransverser = Potential::has_getForceTransverser<MyPotential>::value;
    constexpr bool hasEnergyTransverser = Potential::has_getForceTransverser<MyPotential>::value;
    constexpr bool hasForceEnergyTransverser = Potential::has_getForceEnergyTransverser<MyPotential>::value;
    if(not hasForceTransverser and not hasEnergyTransverser and not hasForceEnergyTransverser){
      auto potname = stringUtils::removePattern(type_name<MyPotential>(), "uammd::");
      sys->log<System::ERROR>("[PairForces] No valid Transverser found in %s.", potname.c_str());
      sys->log<System::ERROR>("[PairForces] At least one of the following must be defined:");
      sys->log<System::CRITICAL>("[PairForces] getForceTransverser, getEnergyTransverser, getForceEnergyTransverser");
    }
    sys->log<System::MESSAGE>("[PairForces] Using Box with size: %g %g %g", box.boxSize.x, box.boxSize.y, box.boxSize.z);
    this->setDelegate(pot.get());
  }


  template<class MyPotential, class NL>
  template<class Transverser>
  void PairForces<MyPotential, NL>::sumTransverser(Transverser &tr, cudaStream_t st){
    this->rcut = pot->getCutOff();
    sys->log<System::DEBUG3>("[PairForces] Using cutOff: %f", this->rcut);
    bool useNeighbourList = true;
    if(box.boxSize.x <=3*rcut and box.boxSize.y <= 3*rcut and box.boxSize.z <= 3*rcut){
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
    //Try to use getForceTransverser, if not present try to use getForceEnergyTransverser, if also not present assume zero force
    sys->log<System::DEBUG1>("[PairForces] Summing forces");
    constexpr bool hasForceTransverser = Potential::has_getForceTransverser<MyPotential>::value;
    constexpr bool hasForceEnergyTransverser = Potential::has_getForceEnergyTransverser<MyPotential>::value;
    if(hasForceTransverser){
      auto ft = Potential::getIfHasForceTransverser<MyPotential>::get(pot, box, pd);
      this->sumTransverser(ft, st);
    }
    else if(hasForceEnergyTransverser){
      auto ft = Potential::getIfHasForceEnergyTransverser<MyPotential>::get(pot, box, pd);
      this->sumTransverser(ft, st);
    }
  }

  template<class MyPotential, class NL>
  real PairForces<MyPotential, NL>::sumEnergy(){
    sys->log<System::DEBUG1>("[PairForces] Summing Energy");
    cudaStream_t st = 0;
    constexpr bool hasEnergyTransverser = Potential::has_getForceTransverser<MyPotential>::value;
    constexpr bool hasForceEnergyTransverser = Potential::has_getForceEnergyTransverser<MyPotential>::value;
    if(hasEnergyTransverser){
      auto et = Potential::getIfHasEnergyTransverser<MyPotential>::get(pot, box, pd);      
      this->sumTransverser(et, st);
    }
    else if(hasForceEnergyTransverser){
      auto et = Potential::getIfHasForceEnergyTransverser<MyPotential>::get(pot, box, pd);
      this->sumTransverser(et, st);
    }
    else{
      return 0;
    }
    return 0;
  }

  template<class MyPotential, class NL>
  real PairForces<MyPotential, NL>::sumForceEnergy(cudaStream_t st){
    sys->log<System::DEBUG1>("[PairForces] Summing Force and Energy");
    auto fet = Potential::getIfHasForceEnergyTransverser<MyPotential>::get(pot, box, pd);
    constexpr bool hasForceEnergyTransverser = Potential::has_getForceEnergyTransverser<MyPotential>::value;
    if(not hasForceEnergyTransverser){
      sumForce(st);
      sumEnergy();
    }
    else{
      this->sumTransverser(fet, st);
    }
    return 0;
  }

  template<class MyPotential, class NL>
  void PairForces<MyPotential, NL>::compute(cudaStream_t st){
    sys->log<System::DEBUG1>("[PairForces] Launching compute transverser");
    auto computeTrans = Potential::getIfHasComputeTransverser<MyPotential>::get(pot, box, pd);
    constexpr bool hasComputeTransverser = Potential::has_getComputeTransverser<MyPotential>::value;
    if(hasComputeTransverser){
      this->sumTransverser(computeTrans, st);
    }
  }

}

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
  PairForces<MyPotential, NL>::PairForces(shared_ptr<ParticleGroup> pg,
					  Parameters par,
					  shared_ptr<MyPotential> pot):
    Interactor(pg,
	       "PairForces/" +
	       stringUtils::removePattern(type_name<NL>(), "uammd::") +
	       "/" +
	       stringUtils::removePattern(type_name<MyPotential>(), "uammd::")),
    box(par.box),
    pot(pot),
    nl(par.nl),
    nb(nullptr)
  {
    constexpr bool hasTransverser = Potential::has_getTransverser<MyPotential>::value;
    if(not hasTransverser){
      auto potname = stringUtils::removePattern(type_name<MyPotential>(), "uammd::");
      sys->log<System::ERROR>("[PairForces] No valid Transverser found in %s.", potname.c_str());
      sys->log<System::ERROR>("[PairForces] A member function called getTransverser must be defined:");      
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
  void PairForces<MyPotential, NL>::sum(Computables comp, cudaStream_t st){
    //Try to use getForceTransverser, if not present try to use getForceEnergyTransverser, if also not present assume zero force
    sys->log<System::DEBUG1>("[PairForces] Summing interaction");
    auto ft = Potential::getIfHasTransverser<MyPotential>::get(pot, comp, box, pd);
    this->sumTransverser(ft, st);
  }
}

/*Raul P. Pelaez 2017. PairForces definition.

  PairForces Module is an interactor that computes short range forces.
  Computes the interaction between neighbour particles (pairs of particles closer tan rcut).
    
  For that, it uses a NeighbourList and computes the force given by Potential for each pair of particles. It sums the force for all neighbours of every particle.

  See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces   for more info.
*/

#ifndef PAIRFORCES_H
#define PAIRFORCES_H

#include"Interactor/Interactor.cuh"
#include"Interactor/NeighbourList/CellList.cuh"
#include"Interactor/NBody.cuh"
#include"third_party/type_names.h"

namespace uammd{
  /*This makes the class valid for any NeighbourList*/
  template<class Potential, class NeighbourList = CellList>
  class PairForces: public Interactor{
  public:
    struct Parameters{
      real rcut;
      Box box;      
    };
    PairForces(shared_ptr<ParticleData> pd,
	       shared_ptr<ParticleGroup> pg,
	       shared_ptr<System> sys,
	       Parameters par,
	       shared_ptr<Potential> pot = std::make_shared<Potential>());
    PairForces(shared_ptr<ParticleData> pd,
	       shared_ptr<System> sys,
	       Parameters par,
	       shared_ptr<Potential> pot = std::make_shared<Potential>()):
      PairForces(pd, std::make_shared<ParticleGroup>(pd, sys, "All"), sys, pot, par){
    }

    ~PairForces(){}
    void sumForce(cudaStream_t st) override;
    real sumEnergy() override;
    //real sumVirial() override{ return 0;}
    
    void print_info(){
      sys->log<System::MESSAGE>("[PairForces] Using: %s Neighbour List.", type_name<NeighbourList>());
      //nl.print();
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


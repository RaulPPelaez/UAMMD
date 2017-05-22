/*Raul P. Pelaez 2017. PairForces definition.

  PairForces Module is an interactor that computes short range forces.
    Computes the interaction between neighbour particles (pairs of particles closer tan rcut).
    
  For that, it uses a NeighbourList and computes the force given by Potential for each pair of particles. It sums the force for all neighbours of every particle.

See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces   for more info.
*/

#ifndef PAIRFORCES_H
#define PAIRFORCES_H
#include"NeighbourList.cuh"
#include"CellList.cuh"
#include"Interactor.h"
#include"globals/defines.h"
#include"globals/globals.h"
#include"misc/Potential.cuh"
#include<functional>
#include<memory>
#include"third_party/type_names.h"

/*This makes the class valid for any NeighbourList*/
template<class NeighbourList, class Potential>
class PairForces: public Interactor{
public:
  /*Default is parameters for gcnf (all system), and LJ potential*/
  PairForces();
  PairForces(real rcut);
  PairForces(real rcut, real3 L, int N);
  ~PairForces(){}
  void sumForce() override;
  real sumEnergy() override;
  real sumVirial() override{return 0;}
  void print_info(){
    std::cerr<<"\t Using: "<<type_name<NeighbourList>()<<" Neighbour List."<<std::endl;
    nl.print();
    std::cerr<<"\t Using: "<<type_name<Potential>()<<" potential."<<std::endl;
	
  }

  template<typename TypeParams = typename Potential::TypeParams>
  void setPotParams(int namei, int namej, TypeParams params){
    pot.setPotParams(namei, namej, params);
  }

private:
  NeighbourList nl;
  Potential pot;
  Vector3 potParams;

  void *cubTempStorage;
  size_t cubTempStorageBytes;
  GPUVector<real> energy;
  
};




#include<PairForces.cu>
  
#endif


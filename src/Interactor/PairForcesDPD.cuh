/*Raul P. Pelaez 2017. PairForcesDPD definition.

  PairForcesDPD Module is an interactor that computes short range forces.
    Computes the interaction between neighbour particles (pairs of particles closer tan rcut).
    
  For that, it uses a NeighbourList and computes the force given by Potential for each pair of particles. It sums the force for all neighbours of every particle.

  Additionally, it computes friction and noise as given by Dissipative Particle Dynamics

TODO

See https://github.com/RaulPPelaez/UAMMD/wiki/Pair-Forces-DPD   for more info.
*/

#ifndef PAIRFORCESDPD_CUH
#define PAIRFORCESDPD_CUH
#include"NeighbourList.cuh"
#include"CellList.cuh"
#include"Interactor.h"
#include"globals/defines.h"
#include"globals/globals.h"
#include"misc/Potential.cuh"
#include<functional>
#include"third_party/type_names.h"

template<class NeighbourList>
class PairForcesDPD: public Interactor{

public:
  /*Default is parameters for gcnf (all system), and LJ potential*/
  PairForcesDPD(std::function<real(real,real)> Ffoo = forceLJ,
		std::function<real(real,real)> Efoo = energyLJ);
  PairForcesDPD(real rcut, real3 L, int N,
		std::function<real(real,real)> Ffoo = forceLJ,
		std::function<real(real,real)> Efoo = energyLJ);
  ~PairForcesDPD(){}
  void sumForce() override;
  real sumEnergy() override{return 0;}
  real sumVirial() override{return 0;}
  void print_info(){
    std::cerr<<"\t Using: "<<type_name<NeighbourList>()<<" Neighbour List."<<std::endl;
    nl.print();
  }

private:
  NeighbourList nl;
  TablePotential pot;
  GPUVector4 sortVel;
  ullint seed;
  real gamma;
  real noiseAmp;
};


#endif
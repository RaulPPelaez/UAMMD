/*Raul P. Pelaez 2017. Three bonded forces, AKA three body springs.

  Joins three particles with an angle bond i---j---k

  Needs an input file containing the bond information as:
  nbonds
  i j k K r0 ang0
  .
  .
  .

  K is the harmonic spring constant, r0 its eq. distance and ang0 the eq angle between ijk.
  The order doesnt matter as long as j is always the central particle in the bond.

  

 */
#ifndef THREEBONDEDFORCES_CUH
#define THREEBONDEDFORCES_CUH

#include"utils/utils.h"
#include"Interactor.h"

#include"globals/globals.h"

#include<cstdint>
#include<memory>
#include<functional>
#include<vector>

#include"third_party/type_names.h"


class ThreeBondedForces: public Interactor{
public:
  struct ThreeBond{
    int i,j,k;
    real r0,kspring,ang;
  };

  explicit ThreeBondedForces(const char * readFile);
  explicit ThreeBondedForces(const char * readFile, real3 L, int N);

  ~ThreeBondedForces();
  
  void sumForce() override;
  real sumEnergy() override;
  real sumVirial() override;
  
private:
  
  uint nbonds;
  Vector<ThreeBond> bondList;
  Vector<uint> bondStart, bondEnd;
  Vector<uint> bondParticleIndex; //Particles with bonds

  int TPP; //Threads per particle
};


#endif

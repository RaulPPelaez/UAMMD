/*Raul P. Pelaez 2017. Three bonded forces, AKA three body springs.

  Joins three particles with an angle bond i---j---k

  Needs an input file containing the bond information as:
  nbonds
  i j k K ang0
  .
  .
  .

  K is the harmonic spring constant, r0 its eq. distance and ang0 the eq angle between ijk.
  The order doesnt matter as long as j is always the central particle in the bond.

  

 */
#ifndef THREEBONDEDFORCES_CUH
#define THREEBONDEDFORCES_CUH

#include"Interactor.cuh"
#include"global/defines.h"
#include<thrust/device_vector.h>
namespace uammd{
  class AngularBondedForces: public Interactor{
  public:
    struct ThreeBond{
      int i,j,k;
      real kspring,ang;
    };
    struct Parameters{
      Box box;
      const char * readFile;
    };
    explicit AngularBondedForces(shared_ptr<ParticleData> pd,
				 shared_ptr<System> sys,
				 Parameters par);

    ~AngularBondedForces();
  
    void sumForce(cudaStream_t st) override;
    real sumEnergy() override;    
  
  private:
  
    int nbonds;
    thrust::device_vector<ThreeBond> bondList;
    thrust::device_vector<int> bondStart, bondEnd;
    thrust::device_vector<int> bondParticleIndex; //Particles with bonds

    int TPP; //Threads per particle

    Box box;
  };

}
#include"AngularBondedForces.cu"
#endif

/*Raul P. Pelaez 2017. NBody Interactor module. All particles interact with all the others.

  See https://github.com/RaulPPelaez/UAMMD/wiki/NBody-Forces for more information

  NBody needs a transverser with the information of what to compute for each particle given all the others.
  You can see example Transversers at the end of "NBody.cuh". If your problem is very similar to one of these transversers, you can inherit from it or simply use it.

Usage:

Use it as any other interactor module (see PairForces.cuh).
Needs a Transverser telling it what to do with each pair of particles. See the end of the file for an example.

  References:
   [1] Fast N-Body Simulation with CUDA. Chapter 31 of GPU Gems 3
*/

#ifndef NBODYFORCES_CUH
#define NBODYFORCES_CUH

#include"Interactor.cuh"
#include"Interactor/NBody.cuh"
#include"ParticleData/ParticleGroup.cuh"

#include"global/defines.h"
#include"third_party/type_names.h"

#include"utils/cxx_utils.h"
#include"Interactor/Potential/PotentialUtils.cuh"
#include"utils/TransverserUtils.cuh"


namespace uammd{

  //NBodyForces uses NBody under the hood, which is used as a neighbour list (through a Transverser).
  //See the wiki for more info on transversers. You can see an example in "NBody.cuh" or "RadialPotential.cuh"

  //In this case, NBodyForces needs a Potential, which can provide transversers to compute force, energy and virial.
  template<class MyPotential>
  class NBodyForces: public Interactor, public ParameterUpdatableDelegate<MyPotential>{
  public:
    struct Parameters{
      Box box;
    };
    NBodyForces(shared_ptr<ParticleData> pd,
		shared_ptr<ParticleGroup> pg,
		shared_ptr<System> sys,
		Parameters par,
	        shared_ptr<MyPotential> pot):
      Interactor(pd, pg, sys,"NBodyForces/"+type_name<MyPotential>()),
      pot(pot),
      box(par.box),
      nb(nullptr){
	this->setDelegate(pot.get());
	nb = std::make_shared<NBody>(pd,pg,sys);
    }
    NBodyForces(shared_ptr<ParticleData> pd,
		shared_ptr<System> sys,
		Parameters par,
		shared_ptr<MyPotential> pot):
      NBodyForces(pd,
		  std::make_shared<ParticleGroup>(pd, sys),
		  sys,
		  par,
		  pot){ }

    ~NBodyForces(){}


    void sumForce(cudaStream_t st) override{
      auto tr = pot->getForceTransverser(box, pd);
      nb->transverse(tr, st);
    }
    real sumEnergy() override{
      auto et = Potential::getIfHasEnergyTransverser<MyPotential>::get(pot, box, pd);
      //If a null transverser has been issued, just return 0
      constexpr bool isnull = std::is_same<decltype(et), BasicNullTransverser>::value;
      if(isnull)
	return 0.0;
      else
	nb->transverse(et, 0);
      return 0.0;
    }

    void print_info(){
      sys->log<System::MESSAGE>("[NBodyForces] Transversing with: %s", type_name<MyPotential>());
    }

  private:

    shared_ptr<MyPotential> pot;
    Box box;
    shared_ptr<NBody> nb;
  };


}

#endif
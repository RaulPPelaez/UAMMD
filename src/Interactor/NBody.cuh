/*Raul P. Pelaez 2017-2021. NBody submodule.

  An NBody is a very lightweight object than can be used to process Transversers with an all-with-all nbody interaction O(N^2).

USAGE:

Create an instance:
NBody nbody(particleData, particleGroup, system);

Use it to process a transverser:

nbody.transverse(myTransverser, cudaStream);

It has a very low memory footprint and a very fast initialization,
  so do not bother storing it, just create it when needed.

See more about transversers and how to implement them in the wiki page[1].

For use outside the UAMMD ecosystem, see NBodyBase.cuh
[1] https://github.com/RaulPPelaez/UAMMD/wiki/Transverser
*/
#ifndef NBODY_CUH
#define NBODY_CUH
#include"global/defines.h"
#include"ParticleData/ParticleGroup.cuh"
#include"Interactor/NBodyBase.cuh"
namespace uammd{
  class NBody{
    shared_ptr<ParticleGroup> pg;
    shared_ptr<ParticleData> pd;
    shared_ptr<System> sys;
    NBodyBase nb;
  public:
    NBody(shared_ptr<ParticleData> pd,
	  shared_ptr<ParticleGroup> pg,
	  shared_ptr<System> sys): pg(pg), pd(pd), sys(sys){
      sys->log<System::DEBUG>("[NBody] Created");
    }

    NBody(shared_ptr<ParticleData> pd):
      NBody(pd, std::make_shared<ParticleGroup>(pd), pd->getSystem()){
    }

    template<class Transverser>
    inline void transverse(Transverser &a_tr, cudaStream_t st = 0){
      int N = pg->getNumberParticles();
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      SFINAE::TransverserAdaptor<Transverser>::prepare(a_tr, pd);
      nb.transverse(pos.begin(), groupIterator, a_tr, N, st);
    }
  };

}


#endif

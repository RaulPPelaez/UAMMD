/*Raul P.Pelaez. 2017-2021. External Forces Module.
  Computes the effect of an external potential acting on each particle independently.
  i.e harmonic confinement, gravity...

  Needs a functor with of the following form:
//External potential acting on each particle independently.
struct HarmonicWall{
real k, zwall;
HarmonicWall(real k, real zwall):k(k), zwall(zwall){}
  //This function will be called for each particle
  //The arguments will be modified according to what was returned by getArrays below
  __device__ ForceEnergyVirial sum(Interactor::Computables comp, real4 pos,// real mass){
    //The decision to compute energy/virial and or force should come from the members of
    // comp (comp.force, comp.energy)
    real3 force = (comp.force or comp.virial)?make_real3(0.0f, 0.0f, -k*(pos.z-zwall)):real3();
    real energy = comp.energy?real(0.5)*k*pow(pos.z-zwall, 2):0;
    real virial = comp.virial?dot(f,make_real3(pos)):0;
    return {force,energy, virial};
  }
    
  auto getArrays(ParticleData* pd){
    auto pos = pd->getPos(access::gpu, access::read);    
    return pos.begin();
    //If more than one property is needed this would be the way to do it:
    //auto mass = pd->getMass(access::gpu, access::read);
    //return std::make_tuple(pos.begin(), mass.begin());
    //In this case the additional arrays must appear as additional arguments in "sum"
  }

ExternalForces will call getArrays expecting a list of arrays (with size pd->getNumParticles() and the order of the particle properties).
It will then read for each particle index from the array list and pass the values to the force/energy functions
So the input parameters of force/energy must be the same as the outputs from getArrays and in the smae order (but without the pointer).

Here is an example in which more properties are needed to compute the force:


struct ReallyComplexExternalForce{
  real drag;
  ReallyComplexExternalForce(real drag):drag(drag){}

  __device__ ForceEnergyVirial sum(Interactor::Computables comp, real4 pos, real3 vel, int id, real mass){
  //...
  //real3 force =...
  //real energy =...
  //real virial =...
  return {force, energy, virial};
  }

  auto getArrays(ParticleData* pd){
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto vel = pd->getVel(access::location::gpu, access::mode::read);
    auto id = pd->getId(access::location::gpu, access::mode::read);
    auto mass = pd->getMass(access::location::gpu, access::mode::read);
    return {pos.begin(), vel.begin(), id.begin(), mass.begin()};
  }


References:
[1] https://www.murrayc.com/permalink/2015/12/05/modern-c-variadic-template-parameters-and-tuples/

*/
#ifndef EXTERNALFORCES_CUH
#define EXTERNALFORCES_CUH
#include"Interactor.cuh"
#include"ParticleData/ParticleGroup.cuh"
#include"global/defines.h"
#include"third_party/type_names.h"
#include"utils/cxx_utils.h"
#include"utils/TransverserUtils.cuh"
#include <initializer_list>
namespace uammd{
  /*Computes the effect of an external potential acting on each particle independently.
    i.e harmonic confinement, gravity...
  */
  template<class Functor>
  class ExternalForces: public Interactor, public ParameterUpdatableDelegate<Functor>{
  public:

    ExternalForces(shared_ptr<ParticleGroup> pg, std::shared_ptr<Functor> tr = std::make_shared<Functor>()):
      Interactor(pg, "ExternalForces/"+type_name<Functor>()), tr(tr){
      //ExternalForces does not care about any parameter update, but the Functor might.
      this->setDelegate(this->tr.get());
    }
    //If no group is provided, a group with all particles is assumed
    ExternalForces(shared_ptr<ParticleData> pd, std::shared_ptr<Functor> tr = std::make_shared<Functor>()):
      ExternalForces(std::make_shared<ParticleGroup>(pd, "All"), tr){
    }

    void sum(Computables comp, cudaStream_t st = 0) override;

  private:
    std::shared_ptr<Functor> tr;
  };

  namespace ExternalForces_ns{
    //These functions can transform several return types gathered from getArrays into a tuple
    template<class ...T> constexpr auto getTuple(std::tuple<T...> tuple){
      return tuple;
    }

    template<class T> constexpr auto getTuple(T* ptr){
      return std::make_tuple(ptr);
    }

    template<class T> constexpr auto getTuple(std::initializer_list<T> ptrlist){
      return std::make_tuple(ptrlist);
    }

  }

  namespace ExternalForces_ns{
    template<class Functor, class ...T, size_t ...Is>
    __device__ inline auto unpackTupleAndCallSum_impl(Functor &f,
						      Interactor::Computables comp,
						      std::tuple<T...> &arrays,
						      int i, //Element of the array to dereference
						      std::index_sequence<Is...>){
      return SFINAE::SumDelegator<Functor>().sum(f, comp, *(std::get<Is>(arrays)+i)...);
    }

    template<class Functor, class ...T>
    __device__ inline auto unpackTupleAndCallSum(Functor &f,
						 Interactor::Computables comp,
						 int i, std::tuple<T...> &arrays){
      constexpr int n= sizeof...(T); //Number of elements in tuple (AKA arguments in () operator in Functor)
      return unpackTupleAndCallSum_impl(f, comp, arrays, i,
					std::make_index_sequence<n>());
    }

    template<class Functor, class ...T>
    __global__ void computeSumGPU(Functor f,
				  int numberParticlesInGroup,
				  ParticleGroup::IndexIterator groupIterator,
				  Interactor::Computables comp,
				  real4* force, real* energy, real* virial,
				  std::tuple<T...>  arrays){
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=numberParticlesInGroup) return;
      const int myParticleIndex = groupIterator[id];
      auto res = unpackTupleAndCallSum(f, comp, myParticleIndex, arrays);
      if(comp.force)  force [myParticleIndex] += make_real4(res.force);
      if(comp.energy) energy[myParticleIndex] += res.energy;
      if(comp.virial) virial[myParticleIndex] += res.virial;
      
    }
  }
  
  template<class Functor>
  void ExternalForces<Functor>::sum(Computables comp, cudaStream_t st){
    sys->log<System::DEBUG1>("[ExternalForces] Computing...");
    int numberParticles = pg->getNumberParticles();
    int blocksize = 128;
    int Nthreads = blocksize<numberParticles?blocksize:numberParticles;
    int Nblocks = numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
    auto force  = comp.force?pd->getForce(access::gpu, access::readwrite).begin():nullptr;
    auto energy = comp.energy?pd->getEnergy(access::gpu, access::readwrite).begin():nullptr;
    auto virial = comp.virial?pd->getVirial(access::gpu, access::readwrite).begin():nullptr;
    auto groupIterator = pg->getIndexIterator(access::location::gpu);
    ExternalForces_ns::computeSumGPU<<<Nblocks, Nthreads, 0, st>>>(*tr, numberParticles,
								groupIterator,
								comp, force, energy, virial,
								ExternalForces_ns::getTuple(tr->getArrays(pd.get())));
  }

}
#endif

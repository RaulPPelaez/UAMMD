/*Raul P. Pelaez 2018. Some utilities for working with Potentials.

  
 */
#ifndef POTENTIALUTILS_CUH
#define POTENTIALUTILS_CUH

#include"utils/cxx_utils.h"
#include"utils/TransverserUtils.cuh"
namespace uammd{
  namespace Potential{
    //This macro defines a struct called has_EnergyTransverser
    SFINAE_DEFINE_HAS_MEMBER(EnergyTransverser)

    //Calling get on this struct will provide the EnergyTransverser of a Potential if it exists, and will return a BasicNullTransverser otherwise
    template<class T, bool hasEnergyTransverser = has_EnergyTransverser<T>::value>
    struct getIfHasEnergyTransverser;

    template<class T>
    struct getIfHasEnergyTransverser<T, true>{
      static decltype(&T::getEnergyTransverser) get(shared_ptr<T> t, Box box, shared_ptr<ParticleData> pd){
	return t.getEnergyTransverser(box, pd);
      }

    };
  
    template<class T>
    struct getIfHasEnergyTransverser<T, false>{
      static BasicNullTransverser get(shared_ptr<T> t, Box box, shared_ptr<ParticleData> pd){
	return BasicNullTransverser();
      }

    };
  }
}

#endif
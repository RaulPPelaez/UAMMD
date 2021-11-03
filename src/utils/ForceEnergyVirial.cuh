#ifndef UAMMD_FORCEENERGYVIRIAL_CUH
#define UAMMD_FORCEENERGYVIRIAL_CUH
#include"global/defines.h"
#include"utils/vector.cuh"
namespace uammd{
  struct ForceEnergyVirial{
    real3 force;
    real energy;
    real virial;
  };
  
  VECATTR ForceEnergyVirial operator +(const ForceEnergyVirial &a, const ForceEnergyVirial &b){
    return {a.force + b.force, a.energy + b.energy, a.virial + b.virial};
  }

  VECATTR  void operator +=(ForceEnergyVirial &a, const ForceEnergyVirial &b){
    a.force += b.force;
    a.energy += b.energy;
    a.virial += b.virial;
  }

}
#endif

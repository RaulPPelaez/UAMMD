/* Raul P. Pelaez 2017-2022. Main UAMMD include.

 */
#ifndef UAMMD_CUH
#define UAMMD_CUH


#include "global/defines.h"
#include "System/System.h"

#ifdef UAMMD_EXTENSIONS
  #ifdef UAMMD_EXTENSIONS_PREAMBLE
    #include UAMMD_EXTENSIONS_PREAMBLE
  #else
    #include "../extensions/preamble.h"
  #endif
#endif

#include "ParticleData/ParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#include "Integrator/Integrator.cuh"
#include "Interactor/Interactor.cuh"
#endif

/* Raul P. Pelaez 2017-2022. Main UAMMD include.

 */
#ifndef UAMMD_CUH
#define UAMMD_CUH

#include "System/System.h"
#include "global/defines.h"

// Helper macro to handle indirect inclusion
#define STRINGIFY(x) #x
#define INCLUDE_FILE(x) STRINGIFY(x)

#ifdef UAMMD_EXTENSIONS
#ifdef UAMMD_EXTENSIONS_PREAMBLE
#include INCLUDE_FILE(UAMMD_EXTENSIONS_PREAMBLE)
#else
#include "../extensions/preamble.h"
#endif
#endif

#include "Integrator/Integrator.cuh"
#include "Interactor/Interactor.cuh"
#include "ParticleData/ParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#endif

/*Raul P. Pelaez 2017-2025. Interactor Base class.
 */

#pragma once

/**
 * @def EXTRA_COMPUTABLES
 * @brief Macro to define additional computable quantities at compile time.
 *
 * This macro allows the user to extend the set of quantities that an
 * Interactor can compute by defining additional fields in the
 * @ref uammd::Interactor::Computables "Computables" structure.
 *
 * It should be defined as a Boost.Preprocessor sequence of identifiers.
 * These identifiers will be expanded into boolean flags in the @ref
 * uammd::Interactor::Computables "Computables" struct.
 *
 * @warning This macro must be defined before including any uammd headers.
 *
 * **Example**
 * @code
 * #define EXTRA_COMPUTABLES (myComputable1)(myComputable2)
 * #include "uammd.cuh"
 * // Now the Interactor::Computables struct passed down to sum() will have
 * // additional boolean fields myComputable1 and myComputable2.
 * @endcode
 */
#ifndef EXTRA_COMPUTABLES
#define EXTRA_COMPUTABLES
#endif

#include "ParticleData/ParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#include "System/System.h"
#include "misc/ParameterUpdatable.h"
#include "third_party/type_names.h"
#include <memory>
#include <third_party/boost/preprocessor.hpp>
#include <third_party/boost/preprocessor/seq/for_each.hpp>
namespace uammd {

/**
 * @class Interactor
 * @brief Interface for interaction modules computing forces, energies, and
 * related quantities.
 *
 * An Interactor acts on a group of particles and defines a computation
 * interface through the ::sum() function. This function can compute various
 * physical quantities based on the interaction model implemented by the derived
 * class.
 *
 * Additionally, because Interactor inherits from ParameterUpdatable, it may
 * also respond to parameter update calls during simulation.
 */

class Interactor : public virtual ParameterUpdatable {
protected:
  string name;                  ///< Name of the interactor, mainly used for logging
  shared_ptr<ParticleData> pd;  ///< Shared pointer to the particle data
  shared_ptr<ParticleGroup> pg; ///< Group of particles the interactor acts on
  shared_ptr<System>
      sys; ///< Access to the simulation System instance. This is just a
           ///< convenience member, since the same instance can be accessed via
           ///< @ref ::ParticleData::getSystem .
  virtual ~Interactor() {
    sys->log<System::DEBUG>("[Interactor] %s Destroyed", name.c_str());
  }

public:
  Interactor(shared_ptr<ParticleData> pd, std::string name = "noName")
      : Interactor(std::make_shared<ParticleGroup>(pd, "All"), name) {}

  /**
   * @brief Constructor using a specific particle group.
   * @param pg Shared pointer to a ParticleGroup.
   * @param name Optional name for the interactor.
   */
  Interactor(shared_ptr<ParticleGroup> pg, std::string name = "noName")
      : pd(pg->getParticleData()), pg(pg), sys(pd->getSystem()), name(name) {
    sys->log<System::MESSAGE>("[Interactor] %s created.", name.c_str());
    sys->log<System::MESSAGE>("[Interactor] Acting on group %s",
                              pg->getName().c_str());
  }

  /**
   * @struct Computables
   * @brief Flags specifying which physical quantities to compute.
   *
   * This structure allows the caller to specify which types of data (force,
   * energy, virial, etc.) should be computed during the ::sum() operation.
   * Additional user-defined quantities can also be included via the
   * @ref EXTRA_COMPUTABLES macro.
   */
  struct Computables {
    bool force = false;  ///< Compute forces on particles.
    bool energy = false; ///< Compute energy contributions.
    bool virial = false; ///< Compute virial contributions.
    bool stress = false; ///< Compute stress contributions.
#define DECLARE_EXTRA_COMPUTABLES(r, data, name)                               \
  bool name = false; ///< Compute #name
    BOOST_PP_SEQ_FOR_EACH(DECLARE_EXTRA_COMPUTABLES, _, EXTRA_COMPUTABLES)
#undef DECLARE_EXTRA_COMPUTABLES
  };
  /**
   * @brief Compute the interaction quantities based on specified Computables.
   * After calling ::sum the relevant particle properties will be updated and
   * can be accessed via ParticleData.
   * @note This method must be implemented by any derived Interactor class.
   *
   * @param comp Struct indicating which quantities should be computed.
   * @param st CUDA stream for computation. All CUDA calls are guaranteed to be
   * synchronous with respect to this stream.
   */
  virtual void sum(Computables comp, cudaStream_t st = 0) = 0;

  std::string getName() {
    return this->name;
  } ///< Get the name of the interactor.
};

} // namespace uammd

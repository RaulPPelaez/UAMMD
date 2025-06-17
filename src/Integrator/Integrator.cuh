/* Raul P. Pelaez 2017-2021. Integrator Base class
   Integrator is an UAMMD base module,
   to create an Integrator module inherit from Integrator and overload the
   virtual methods. An Integrator has the ability to move the simulation one
   step forward in time.

   For that, it can use any number of Interactors.

   Integrator can also hold a list of references to ParameterUpdatable derived
   objects. Adding an Interactor via addInteractor() will also add it as a
   ParameterUpdatable.

   See the related wiki page for more information.

   USAGE:
   //Say "integrator" is an instance of an Integrator derived class (such as
   BD::EulerMaruyama or VerletNVE).

   //Forward the simulation to the next time step:
   integrator.forwardTime();

   //Add an Interactor derived object (as a shared_ptr) to the integrator:
   integrator.addInteractor(an_interactor);
   //Add a ParameterUpdatable derived object (asa shared_ptr) to the integrator:
   //Note that interactors are automatically added as updatables as well, so
   there is no need to manually add them as updatables.
   integrator.addUpdatable(an_updatable);

   //Get a list of Interactors that have been added to the Integrator:
   auto interactors = integrator.getInteractors();
   //Get a list of ParameterUpdatables that have been added to the Integrator
   (note that this includes Interactors): auto updatables =
   integrator.getUpdatables();
   //Note that calling any "update" method for both the interactors and
   updatables lists will duplicate calls to the update methods of the
   interactors.
   //If the Integrator needs to update parameters it should do so using only the
   list provided by addUpdatables().



 */
#pragma once
#include "Interactor/Interactor.cuh"
#include "ParticleData/ParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#include "System/System.h"
#include <memory>
#include <set>
#include <vector>

namespace uammd {

/**
 * @class Integrator
 * @brief Base class for implementing time integration in particle simulations.
 *
 * The Integrator class defines a standard interface for advancing
 * the state of a particle system in time. It manages interactions,
 * energy contributions, and parameter updates.
 */
class Integrator {
protected:
  std::string name;                 ///< Name of the integrator
  std::shared_ptr<ParticleData> pd; ///< Shared pointer to the particle data
  std::shared_ptr<ParticleGroup>
      pg;                      ///< Group of particles the integrator acts on
  std::shared_ptr<System> sys; ///< Simulation system
  std::vector<std::shared_ptr<Interactor>>
      interactors; ///< List of interactors added to the integrator
  std::set<std::shared_ptr<ParameterUpdatable>>
      updatables; ///< Set of parameter updatables, including interactors

  virtual ~Integrator() {
    sys->log<System::DEBUG>("[Integrator] %s Destroyed", name.c_str());
  }

public:
  Integrator(std::shared_ptr<ParticleData> pd, std::string name = "noName")
      : Integrator(std::make_shared<ParticleGroup>(pd, "All"), name) {}

  /**
   * @brief Constructor that uses a specific ParameterUpdatable.
   * @param i_pg Pointer to a ParticleGroup.
   * @param name Optional name for the integrator.
   */
  Integrator(std::shared_ptr<ParticleGroup> i_pg, std::string name = "noName")
      : pd(i_pg->getParticleData()), pg(i_pg),
        sys(i_pg->getParticleData()->getSystem()), name(name) {
    sys->log<System::MESSAGE>("[Integrator] %s created.", name.c_str());
    sys->log<System::MESSAGE>("[Integrator] Acting on group %s",
                              pg->getName().c_str());
  }

  /**
   * @brief Advance the simulation to the next time step.
   */
  virtual void forwardTime() = 0;

  /**
   * @brief Compute the energy contribution from the integrator.
   *
   * This typically includes kinetic energy.
   *
   * @return Energy contribution as a real number.
   */
  virtual real sumEnergy() { return 0.0; }

  /**
   * @brief Add an Interactor to the Integrator.
   *
   * The interactor is also added as a ParameterUpdatable.
   *
   * @param an_interactor Shared pointer to the interactor.
   */
  void addInteractor(std::shared_ptr<Interactor> an_interactor) {
    sys->log<System::MESSAGE>("[%s] Adding Interactor %s...", name.c_str(),
                              an_interactor->getName().c_str());
    interactors.emplace_back(an_interactor);
    this->addUpdatable(an_interactor);
  }

  /**
   * @brief Retrieve the list of interactors added to the integrator.
   *
   * @return Vector of shared pointers to Interactors.
   */
  auto getInteractors() { return interactors; }

  /**
   * @brief Add a ParameterUpdatable to the integrator.
   *
   * @param an_updatable Shared pointer to a ParameterUpdatable.
   */
  void addUpdatable(std::shared_ptr<ParameterUpdatable> an_updatable) {
    sys->log<System::MESSAGE>("[%s] Adding updatable", name.c_str());
    updatables.emplace(an_updatable);
  }

  /**
   * @brief Get the list of ParameterUpdatables added to the integrator.
   *
   * Includes both explicitly added updatables and the interactors.
   *
   * @return Vector of shared pointers to ParameterUpdatables.
   */
  auto getUpdatables() {
    return std::vector<std::shared_ptr<ParameterUpdatable>>(updatables.begin(),
                                                            updatables.end());
  }
};

} // namespace uammd

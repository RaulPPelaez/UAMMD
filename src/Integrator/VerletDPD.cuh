#pragma once
// Raul P. Pelaez 2025. Verlet DPD Integrator module.
#include "Integrator/Integrator.cuh"
#include "Integrator/DPD/DissipationKernels.cuh"
namespace uammd {
namespace dpd {
/**
 * @brief Concept for Dissipation Kernels in DPD simulations.
 *
 * This concept defines the requirements for any object that can act as a force
 * functor in DPD.
 *
 * The functor must be callable from device code with the following signature:
 *
 * @code
 * __device__ real3 operator()(int i, int j, real3 rij, real3 vij, real cutoff,
 * Saru& rng);
 * @endcode
 *
 * where:
 * - `i`, `j`: particle indices.
 * - `rij`: relative position vector between particles `i` and `j`,
 * - `vij`: relative velocity vector between particles `i` and `j`,
 * - `cutoff`: cutoff radius for interactions,
 * - `rng`: a Saru random number generator instance, seeded identically for the
 * (i,j) and (j,i) pairs
 *
 * The call must return a @ref real3 representing the total force vector applied
 * to particle `i` due to particle `j`.
 *
 * @tparam T The type to be checked for compatibility as a force functor.
 */
template <typename T>
concept DissipationKernel = requires(const T &dk, int i, int j, real3 rij,
                                     real3 vij, real cutoff, Saru &rng) {
  { dk(i, j, rij, vij, cutoff, rng) } -> std::convertible_to<real3>;
};
// clang-format off

// clang-format off
/**
 * @brief Modified Velocity Verlet algorithm for Dissipative Particle Dynamics.
 *
 * This is a modified Velocity Verlet (VV) algorithm that addresses the temporal
 * misalignment which arises in the standard VV approach when the force depends
 * on the velocity. The modified algorithm proceeds as follows:
 *
 * @f[
 * &\mathbf{v}_{i}^{n+1/2} = \mathbf{v}_{i}^{n} + \frac{\Delta t}{2} \mathbf{a}_{i}^{n} \\
 * &\mathbf{r}_{i}^{n+1}   = \mathbf{r}_{i}^{n} + \Delta t \mathbf{v}_{i}^{n+1/2} \\
 * &\widetilde{\mathbf{v}}_{i}^{n+1/2} = \mathbf{v}_{i}^{n} + \lambda \frac{\Delta t}{2} \mathbf{a}_{i}^{n+1} \\
 * &\mathbf{a}_{i}^{n+1} = m_i^{-1}\sum_{j\in\mathcal{N}}\mathbf{f}\left(\mathbf{r}_{ij}^{n+1}, \widetilde{\mathbf{v}}^{n+1/2}_{ij}\right) \\
 * &\mathbf{v}_{i}^{n+1} = \mathbf{v}_{i}^{n+1/2} + \frac{\Delta t}{2} \mathbf{a}_{i}^{n+1}
 * @f]
 *
 * where @f$\mathbf{r}_{i}^{n}@f$ and @f$\mathbf{v}_{i}^{n}@f$ are the positions and velocities
 * of particle @f$i@f$ at time step @f$n@f$, @f$\mathbf{a}_{i}^{n}@f$ is the acceleration at time step @f$n@f$,
 * and @f$\mathbf{f}@f$ is a function that computes the force on particle @f$i@f$ due to particle @f$j@f$ in
 * its neighborhood, @f$\mathcal{N}@f$, according to the chosen @ref DissipationKernel.
 *
 * @remark In practice, the acceleration at each step might include the action of an arbitrary number
 *         of additional @ref uammd::Interactor "Interactors". This @ref uammd::Integrator "Integrator"
 *         will always include the action of a @ref uammd::dpd::DissipationKernel "DissipationKernel"
 *         , which is a DPD-like dissipative force.
 *
 * The parameter @f$\lambda \in [0, 1]@f$ is an empirical coefficient accounting for
 * stochastic effects and must be tuned for each specific system. The standard Velocity Verlet
 * algorithm is recovered with @f$\lambda = 0.5@f$. According to [1], a value of
 * @f$\lambda = 0.65@f$ is optimal for systems with density @f$\rho = 3.0@f$ and
 * noise amplitude @f$\sigma = 3.0@f$.
 *
 * @tparam Kernel A type that implements the @ref DissipationKernel concept.
 *                See @ref DefaultDissipation for an example implementation.
 * @par References
 * [1] R.D. Groot, P.B. Warren, *Dissipative particle dynamics: bridging the gap
 * between atomistic and mesoscopic simulation*, J. Chem. Phys. 107(11), 4423–4435 (1997).
 */
// clang-format on
template <DissipationKernel Kernel> class Verlet : public Integrator {
public:
  /**
   * @brief Parameters for the Verlet integrator.
   */
  struct Parameters {
    /**
     * @brief DPD velocity temporal scaling parameter @f$\lambda@f$.
     *
     * A value typically between 0 and 1. It controls how velocity is scaled
     * during time integration. For DPD-like thermostats, this is used to blend
     * conservative and stochastic contributions in velocity updates. 0.5
     * recovers Velocity Verlet.
     */
    real lambda = 0.5;

    real dt = 0; ///< Time step for the simulation.

    bool is2D = false; ///< Whether the simulation is in 2D or 3D.

    /**
     * Mass of the particles in simulation units. A negative value indicates
     * that the mass for each particle will be taken from the particle data.
     */
    real mass = -1;
    real temperature = 1.0; ///< Temperature of the solvent in units of energy.
                            ///< This is @f$k_BT@f$ in the formulas.
    std::shared_ptr<Kernel> dissipation; ///< Dissipation kernel to be used in the simulation.
  };

  Verlet(shared_ptr<ParticleData> pd, Parameters par)
      : Verlet(std::make_shared<ParticleGroup>(pd, "All"), par) {}

  /**
   * @brief Constructor using a specific particle group.
   *
   * @param pg Shared pointer to a ParticleGroup.
   * @param par Parameters for the Verlet integrator.
   */
  Verlet(shared_ptr<ParticleGroup> pg, Parameters par);

  /**
   * @brief Advance the simulation to the next time step.
   */
  void forwardTime() override;

  /**
   * @brief Compute the energy contribution from the integrator.
   *
   * In the case of DPD, this is computed as the kinetic energy of the particles
   *
   * @return Energy contribution as a real number.
   */
  real sumEnergy() override;
};
} // namespace dpd
} // namespace uammd

#include "VerletDPD.cu"

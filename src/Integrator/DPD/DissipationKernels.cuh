// Raul P. Pelaez 2025. Dissipation kernels compatible with DPD Integrators
#pragma once
#include "misc/ParameterUpdatable.h"
#include "uammd.cuh"
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
/**
 * @brief Default implementation of a DissipationKernel for DPD simulations.
 * Computes the dissipative force between particles as:
 * @f[
 * \vec{F} = \vec{F^c} + \vec{F^d} + \vec{F^r}.
 * @f]
 * Where the three forces are expressed as,
 * @f[
 *  \vec{F^c}_{ij} &=\omega(r_{ij})\hat{\vec{\ppos}}_{ij}\\
 *  \vec{F^d}_{ij} &=-\gamma\omega^2(r_{ij})(\vec{\pvel}_{ij}\cdot\vec{\ppos}_{ij})\hat{\vec{\ppos}}_{ij}\\
 *  \vec{F^r}_{ij} &=\sqrt{2\xi\kT}\omega(r_{ij})\widetilde{W}_{ij}\hat{\vec{\ppos}}_{ij}
 * @f]
 * Where @f$\vec{\pvel}_{ij} = \vec{\pvel}_j - \vec{\pvel}_i@f$ is the relative
 * velocity between particles @f$i@f$ and @f$j@f$. Here @f$\xi@f$ represents a
 * friction coefficient and is related to the random force strength via
 * fluctuation-dissipation balance in a familiar way [1]. In general @f$\xi@f$ can
 * be considered to be a tensorial quantity and even derived from atomistic
 * simulations using dynamic coarse graining theory. The factor
 * @f$\widetilde{W}_{ij}@f$ is different from the one in LD in that it affects
 * pairs of particles (instead of each individual one), it also represents a
 * Gaussian random number with zero mean and unit standard deviation, but must be
 * chosen independently for each pair while ensuring symmetry so that
 * @f$\widetilde{W}_{ij} = \widetilde{W}_{ji}@f$.
 * The weight function @f$\omega(r)@f$ is a soft repulsive force usually defined
 as
 *
 * where the weight function @f$\omega(r)@f$ is defined as:
 * @f[
 *  \omega(r) =  \begin{cases}
 *  \alpha\left(1-\dfrac{r}{r_{c}}\right) & r<r_{c}\\
 *  0 & r\ge r_{c}
 *  \end{cases}
 * @f]
 * Where @f$r_{c}@f$ is a cut-off distance. The strength parameter,
 @f$\alpha@f$,
 * can in principle be different for each pair of particles, @f$i@f$ - @f$j@f$,
 but for this dissipation makes it the same for every pair.
 *
 * ## References
 * [1] Statistical Mechanics of Dissipative Particle Dynamics. P Español and P
 Warren 1995. https://doi.org/10.1209/0295-5075/30/4/001
 */
// clang-format on
struct DefaultDissipation : public ParameterUpdatable {
  real A;
  real gamma;
  real sigma; // Random force strength
  real temperature, dt;
  /**
   * @brief Constructs a DefaultDissipation object.
   *
   * @param A The strength of the conservative force, @f$\alpha@f$.
   * @param gamma The friction coefficient, @f$\gamma@f$.
   * @param temperature The temperature of the system in energy units
   * @f$k_BT@f$.
   * @param dt The time step for the simulation.
   */
  DefaultDissipation(real A, real gamma, real temperature, real dt)
      : A(A), gamma(gamma), temperature(temperature), dt(dt) {
    this->sigma = sqrt(2.0 * temperature / dt);
    System::log<System::MESSAGE>(
        "[DPDDefaultDissipation] Created with A: %f, gamma: "
        "%f, temperature: %f, dt: %f",
        A, gamma, temperature, dt);
  }

  /**
   * @brief Operator to compute the dissipative forces between two particles.
   *
   * @param i Index of the first particle.
   * @param j Index of the second particle.
   * @param rij Relative position vector between particles i and j.
   * @param vij Relative velocity vector between particles i and j.
   * @param cutoff Cutoff radius for the interaction.
   * @param rng Random number generator instance.
   */
  __device__ real3 operator()(int i, int j, real3 rij, real3 vij, real cutoff,
                              Saru &rng) const {
    const real rmod = sqrt(dot(rij, rij));
    const real invrmod = real(1.0) / rmod;
    const auto g = gamma;
    // This weight function is arbitrary as long as wd = wr*wr
    const real wr = real(1.0) - rmod / cutoff;
    const real Fc = A * wr * invrmod;
    const real wd = wr * wr; // Wd must be such as wd = wr^2 to ensure
                             // fluctuation dissipation balance
    const auto Fd = -g * wd * invrmod * invrmod * dot(rij, vij);
    const real Fr = rng.gf(real(0.0), sigma * sqrt(g) * wr * invrmod).x;
    return (Fc + Fd + Fr) * rij;
  }
  /**
   * @brief Updates the temperature of the dissipative force.
   *
   * This method recalculates the random force strength based on the new
   * temperature and the current time step.
   *
   * @param newTemp The new temperature in energy units kT.
   */
  virtual void updateTemperature(real newTemp) override {
    temperature = newTemp;
    sigma = sqrt(2.0 * temperature) / sqrt(dt);
  }

  /**
   * @brief Updates the time step for the dissipative force.
   *
   * This method recalculates the random force strength based on the current
   * temperature and the new time step.
   *
   * @param newdt The new time step for the simulation.
   */
  virtual void updateTimeStep(real newdt) override {
    dt = newdt;
    sigma = sqrt(2.0 * temperature) / sqrt(dt);
  }
};
// clang-format off
/**
 * @brief TransversalDissipation implements a DPD-like dissipative force with
 * both parallel and perpendicular components with respect to the particle
 * pairs.
 *
 * Computes the dissipative force between particles as:
 * @f[
 * \vec{F} = \vec{F^c} + \vec{F^d} + \vec{F^r}.
 * @f]
 * Where the three forces are expressed as,
 * @f[
 *  \vec{F^c}_{ij} &= \omega(r_{ij})\hat{\vec{\ppos}}_{ij}\\
 *  \vec{F^d}_{ij} &= -\left[\gamma_{\perp}\mathbb{I} +(\gamma_{\parallel} - \gamma_{\perp})\hat{\vec{\ppos}}_{ij}\otimes\hat{\vec{\ppos}}_{ij}\right]\cdot\vec{v}_{ij} \omega(r_{ij})^2\\
 *  \vec{F^r}_{ij} &= \sqrt{2\kT dt} \left(\tilde{A} d\vec{\bar{W}}_{ij} + \frac{1}{3}\tilde{B} tr(\vec{W}_{ij})\mathbb{I}\right) \hat{\vec{\ppos}}_{ij}
 * @f]
 * Here, @f$\vec{v}_{ij} = \vec{v}_j - \vec{v}_i@f$ is the relative velocity between
 * particles @f$i@f$ and @f$j@f$, @f$\mathbb{I}@f$ is the identity matrix, and
 * @f$\hat{\vec{\ppos}}_{ij} = \vec{r}_{ij}/|\vec{r}_{ij}|@f$ is the unit vector
 * along the vector connecting particles @f$i@f$ and @f$j@f$. The parameters
 * @f$\gamma_{\parallel}@f$ and @f$\gamma_{\perp}@f$ represent the parallel and
 * perpendicular friction coefficients, respectively.
 * The tensor @f$\vec{W}_{ij}^\alpha\beta@f$ is a random tensor with zero mean
 * and variance given by
 * @f[
 *  \langle W_{ij}^{\alpha\beta} W_{kl}^{\gamma\delta} \rangle =
 * (\delta_{ik}\delta_{jl} +
 * \delta_{il}\delta_{jk})\delta^{\alpha\gamma}\delta^{\beta\delta}.
 * @f]
 * Additionally, the random tensor must be symmetric, i.e.,
 * @f$W_{ij}^{\alpha\beta} = W_{ji}^{\alpha\beta}@f$. The symmetrized random
 * tensor is defined as:
 * @f[
 *  \bar{W}_{ij}^{\alpha\beta} = \frac{1}{2}(W_{ij}^{\alpha\beta} +
 * W_{ji}^{\alpha\beta}) - \frac{1}{3} tr(W_{ij})\delta^{\alpha\beta}
 * @f]
 * The parameters @f$\tilde{A}@f$ and @f$\tilde{B}@f$ are defined as:
 * @f[
 *  \tilde{A}(r_{ij}) &= \sqrt{2\gamma_{\perp}}\omega(r_{ij}), \quad
 * \tilde{B}(r_{ij}) = \sqrt{3\gamma_{\parallel} -
 * 2\gamma_{\perp}}\omega(r_{ij})\\
 * @f]
 * The weight function @f$\omega(r)@f$ is a soft repulsive force defined as
 * @f[
 *  \omega(r) =  \begin{cases}
 *  \alpha\left(1-\dfrac{r}{r_{c}}\right) & r<r_{c}\\
 *  0 & r\ge r_{c}
 *  \end{cases}
 * @f]
 * @note The parameters @f$\gamma_{\parallel}@f$ and @f$\gamma_{\perp}@f$ must
 * satisfy the condition
 *       @f$3\gamma_{\parallel} \geq 2\gamma_{\perp}@f$ to ensure
 * stability of the algorithm.
 *
 * ## References
 *
 * [1] Mori–Zwanzig formalism as a practical computational tool. Hijón, Español,
 * et al. 2010  https://doi.org/10.1039/B902479B
 */
// clang-format on
struct TransversalDissipation : public ParameterUpdatable {
  real g_par, g_perp;
  real sigma; // Random force strength, must be such as sigma =
              // sqrt(2*kT*gamma)/sqrt(dt)
  real temperature, dt;
  real A;

  /**
   * @brief Constructs a TransversalDissipation object.
   *
   * @param g_par The parallel friction coefficient, @f$\gamma_{\parallel}@f$.
   * @param g_perp The perpendicular friction coefficient, @f$\gamma_{\perp}@f$.
   * @param A The strength, @f$\alpha@f$, of the conservative force.
   * @param temperature The temperature of the system in energy units
   * @f$k_BT@f$.
   * @param dt The time step for the simulation.
   */
  TransversalDissipation(real g_par, real g_perp, real A, real temperature,
                         real dt)
      : g_par(g_par), g_perp(g_perp), temperature(temperature), dt(dt), A(A) {
    this->sigma = sqrt(2.0 * temperature) / sqrt(dt);
    if (g_par < (2.0 / 3.0) * g_perp) {
      throw std::runtime_error("[TransversalDissipation] g_par must be greater "
                               "than 4/3 * g_perp, found g_par: " +
                               std::to_string(g_par) +
                               " g_perp: " + std::to_string(g_perp));
    }
    if (g_perp < 0) {
      throw std::runtime_error(
          "[TransversalDissipation] g_perp must be non-negative");
    }
  }

  /**
   * @brief Operator to compute the dissipative forces between two particles.
   *
   * @param i Index of the first particle.
   * @param j Index of the second particle.
   * @param rij Relative position vector between particles i and j.
   * @param vij Relative velocity vector between particles i and j.
   * @param cutoff Cutoff radius for the interaction, @f$r_c@f$.
   * @param rng Random number generator instance.
   */
  __device__ __host__ auto operator()(int i, int j, real3 rij, real3 vij,
                                      real cutoff, Saru &rng) const {
    const real rmod = sqrt(dot(rij, rij));
    const real wr = real(1.0) - rmod / cutoff;
    const auto eij = rij / rmod;
    const auto Fc = A * wr * eij;
    const auto g_par_r = g_par * wr * wr;
    const auto g_perp_r = g_perp * wr * wr;
    const auto Fd = this->dissipative(eij, vij, g_par_r, g_perp_r);
    const auto Fr = this->fluctuation(eij, rng, g_par_r, g_perp_r);
    return Fc + Fd + Fr;
  }

  /**
   * @brief Updates the temperature of the dissipative force.
   *
   * This method recalculates the random force strength based on the new
   * temperature and the current time step.
   *
   * @param newTemp The new temperature in energy units kT.
   */
  virtual void updateTemperature(real newTemp) override {
    temperature = newTemp;
    sigma = sqrt(2.0 * temperature) / sqrt(dt);
  }

  /**
   * @brief Updates the time step for the dissipative force.
   *
   * This method recalculates the random force strength based on the current
   * temperature and the new time step.
   *
   * @param newdt The new time step for the simulation.
   */
  virtual void updateTimeStep(real newdt) override {
    dt = newdt;
    sigma = sqrt(2.0 * temperature) / sqrt(dt);
  }

private:
  __device__ __host__ real3 dissipative(real3 eij, real3 vij, real g_par,
                                        real g_perp) const {
    // (eij\dyadic eij )\dot vij
    const auto vdyadic = make_real3(
        vij.x * eij.x * eij.x + vij.y * eij.x * eij.y + vij.z * eij.x * eij.z,
        vij.x * eij.y * eij.x + vij.y * eij.y * eij.y + vij.z * eij.y * eij.z,
        vij.x * eij.z * eij.x + vij.y * eij.z * eij.y + vij.z * eij.z * eij.z);
    const auto videntity = vij;
    const auto gv = g_perp * videntity + (g_par - g_perp) * vdyadic;
    return -gv;
  }

  // TODO: This could be optimized
  __device__ __host__ real3 fluctuation(real3 eij, Saru &rng, real g_par,
                                        real g_perp) const {
    const auto A = g_perp;
    const auto B = g_par - g_perp;
    const auto Atil = sqrt(2 * A);
    const auto Btil = sqrt(3 * B - A);
    const auto noiseX = make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
    const auto noiseY = make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
    const auto noiseZ = make_real3(rng.gf(0, 1), rng.gf(0, 1).x);
    const auto trNoise = real(1.0 / 3.0) * (noiseX.x + noiseY.y + noiseZ.z);
    const auto noiseA =
        real(0.5) *
            (make_real3(
                (noiseX.x + noiseX.x) * eij.x + (noiseX.y + noiseY.x) * eij.y +
                    (noiseX.z + noiseZ.x) * eij.z,
                (noiseY.x + noiseX.y) * eij.x + (noiseY.y + noiseY.y) * eij.y +
                    (noiseY.z + noiseZ.y) * eij.z,
                (noiseZ.x + noiseX.z) * eij.x + (noiseZ.y + noiseY.z) * eij.y +
                    (noiseZ.z + noiseZ.z) * eij.z)) -
        trNoise * eij;
    const auto noiseB = trNoise * eij;
    const auto fluc = sigma * (Atil * noiseA + Btil * noiseB);
    return fluc;
  }
};

} // namespace dpd
} // namespace uammd

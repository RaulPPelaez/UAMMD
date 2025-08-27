// clang-format off
/**
 * @file DPStokesSlab.cuh
 *
 * In the Doubly periodic Stokes geometry (DPStokes), an incompressible fluid
 * exists in a domain which is periodic in the plane and open in the third direction.
 * Contrary to the Quasi2D regime, in DPStokes particles are free to move in any
 * direction (i.e. they are not confined to a plane).
 *
 * The DPStokes solver (described in detail in Raul's manuscript @ref ref1 "[1]")
 * distinguishes between three different regimes:
 * - **Fully open**: the fluid is bounded at infinity.
 * - **A no-slip wall** at the bottom of the domain.
 * - **A no-slip wall** at top and bottom (slit channel).
 *
 * When there are no walls, a virtual domain size exists in the z direction
 * that must contain all the force exerted by the particles to the fluid.
 * A similar thing happens when there is a wall only at the bottom.
 * In all cases, the domain in z is such that \f$z \in (-H/2, H/2)\f$.
 *
 * The BM kernel (see @ref IBM) is used for spreading and interpolating in this module.
 * It can handle both particle forces and torques. The BM kernel is defined as:
 *
 * \f[
 * \phi_{BM}(r,\{\alpha, \beta, w\}) =
 * \begin{cases}
 * \frac{1}{S}\exp\left[\beta\left(\sqrt{1-(r/(h\alpha))^2}-1\right)\right] & |r|/(hw/2)\le 1 \\
 * 0 & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * where \f$h\f$ is the size of a grid cell in the plane.
 *
 * Notes:
 * - Typically one would set \f$\alpha = w/2\f$, however it can be useful to set them separately.
 * - It can never happen that \f$\alpha > w/2\f$, since that would result in a complex number.
 * - The width of the kernel (\f$\beta\f$) is related to the hydrodynamic kernel,
 *   while its support (\f$\alpha, w\f$) and the size of a grid cell in the plane (\f$h\f$)
 *   are set according to a certain tolerance.
 *
 * There are some basic heuristics to choose the optimal parameters for the kernel
 * depending on whether particle forces and torques or just forces are applied.
 * The current implementation does not choose these automatically, so you must
 * explicitly provide them.
 *
 * Recommended parameters:
 *
 * \rst
 *  +------------------------------------------------------------------+--------------------------------------------+
 *  | .. list-table:: Applying both forces (M) and torques (D).        |    .. list-table:: Applying only forces(M) |
 *  |   :header-rows: 1                                                |           :header-rows: 1                  |
 *  |                                                                  |                                            |
 *  |   * - :math:`w_M(=\!w_D)`                                        |           * - :math:`w_M`                  |
 *  |     - 5                                                          |             - 4                            |
 *  |     - 6                                                          |             - 5                            |
 *  |   * - :math:`a/h`                                                |             - 6                            |
 *  |     - 1.560                                                      |           * - :math:`a/h`                  |
 *  |     - 1.731                                                      |             - 1.205                        |
 *  |   * - :math:`\beta_M/w_M`                                        |             - 1.244                        |
 *  |     - 1.305                                                      |             - 1.554                        |
 *  |     - 1.327                                                      |           * - :math:`\beta_M/w_M`          |
 *  |   * - :math:`\beta_D/w_D`                                        |             - 1.785                        |
 *  |     - 2.232                                                      |             - 1.886                        |
 *  |     - 2.216                                                      |             - 1.714                        |
 *  |   * - :math:`\% error_M`                                         |           * - :math:`\% error_M`           |
 *  |     - 0.897                                                      |             - 0.370                        |
 *  |     - 0.151                                                      |             - 0.055                        |
 *  |   * - :math:`\% error_D`                                         |             - 0.021                        |
 *  |     - 0.810                                                      |                                            |
 *  |     - 0.212                                                      |                                            |
 *  |                                                                  |                                            |
 *  +------------------------------------------------------------------+--------------------------------------------+
 * \endrst
 *
 *
 * Additionally, the number of cells in the z direction is chosen such that
 * the largest cell size is \f$h\f$, which requires:
 *
 * \f[
 * n_z = \frac{\pi H}{h}
 * \f]
 *
 * Currently there is no efficient way to compute fluctuations for BDHI, however,
 * the @refuammd{Integrator} includes them using the @ref Lanczos algorithm.
 * Testing shows that the hydrodynamic screening caused by the walls allows
 * Lanczos to converge fast and independently of the number of particles.
 *
 * Thermal drift must also be included (since the resulting mobility depends
 * on the height), which is computed via Random Finite Differences.
 *
 * **The DPStokes solver comes in two different forms:**
 *  * As an independent solver in the class @refuammd{DPStokesSlab_ns::DPStokes}
 *  * As an @refuammd{Integrator} (which uses the solver under the hood) in the class @refuammd{DPStokesSlab_ns::DPStokesIntegrator}.
 *
 * The solver can be used to compute the hydrodynamic displacements of a group of particles with some forces and/or torques acting on them, i.e applying the mobility operator.
 *
 * The @refuammd{Integrator} is able to carry out @ref BDHI simulations by including fluctuations.
 *
 * @section references References
 * @anchor ref1
 * [1] R. P. Pelaez, *Complex fluids in the GPU era*. PhD thesis manuscript, 2022.
 * URL: https://github.com/RaulPPelaez/tesis/raw/main/manuscript.pdf
 */
// clang-format on
// Raul P. Pelaez 2019-2025. Spectral/Chebyshev Doubly Periodic Stokes solver.
#ifndef DOUBLYPERIODIC_STOKES_CUH
#define DOUBLYPERIODIC_STOKES_CUH
#include "Integrator/Integrator.cuh"
#include "StokesSlab/Correction.cuh"
#include "StokesSlab/FastChebyshevTransform.cuh"
#include "StokesSlab/spreadInterp.cuh"
#include "StokesSlab/utils.cuh"
#include "System/System.h"
#include "global/defines.h"
#include "misc/ChevyshevUtils.cuh"
#include "misc/LanczosAlgorithm.cuh"
#include "misc/LanczosAlgorithm/MatrixDot.h"
#include "uammd.cuh"
#include "utils/utils.h"
#include <memory>
#include <stdexcept>
#include <thrust/functional.h>

namespace uammd {
namespace DPStokesSlab_ns {
namespace detail {
template <class ComplexContainer>
auto getZeroModeChebCoeff(const ComplexContainer &fourierChebGridData, int3 n) {
  std::vector<complex> chebCoeff(n.z);
  for (int i = 0; i < n.z; i++) {
    chebCoeff[i] = fourierChebGridData[(n.x / 2 + 1) * n.y * i] / (n.x * n.y);
  }
  return chebCoeff;
}
} // namespace detail

/**
 * @class DPStokes
 * @brief Implements a doubly periodic Stokes solver.
 */
class DPStokes {
public:
  using Grid = chebyshev::doublyperiodic::Grid;
  using WallMode = WallMode;
  /**
   * @brief Parameters for the DPStokes solver. These are shared with
   * DPStokesIntegrator.
   */
  struct Parameters {
    int nx;      ///< Number of grid points in the x direction
    int ny;      ///< Number of grid points in the y direction
    int nz = -1; ///< Number of grid points in the z direction, -1 means that it
                 ///< will be autocomputed if not present
    real viscosity; ///< Fluid viscosity
    real Lx;        ///< Domain size in the x direction
    real Ly;        ///< Domain size in the y direction
    real H;         ///< Domain size in the z direction, goes from -H/2 to H/2
    real w;   ///< Support distance, in units of the grid cell size, of the
              ///< monopole kernel function.
    real w_d; ///< Support distance, in units of the grid cell size, of the
              ///< dipole kernel function.
    real3 beta = {-1.0, -1.0,
                  -1.0}; ///< Width of the monopole kernel function in each
                         ///< direction. At least X should be set.
    real3 beta_d = {-1.0, -1.0,
                    -1.0}; ///< Width of the dipole kernel function in each
                           ///< direction. At least X should be set.
    real alpha =
        -1; ///< Weight of the monopole kernel function. Typically set to w/2
    real alpha_d =
        -1; ///< Weight of the dipole kernel function. Typically set to w_d/2
    WallMode mode =
        WallMode::none; ///< Wall mode, can be either none, bottom or slit
  };
  /**
   * @brief Constructs a DPStokes object with the given parameters.
   * @param par The parameters for the DPStokes solver.
   */
  DPStokes(Parameters par);

  ~DPStokes() { System::log<System::MESSAGE>("[DPStokes] Destroyed"); }

  /**
   * @brief Computes the hydrodynamic displacements (velocities) resulting from
   the forces acting on a group of positions.

   * @param pos Iterator containing the positions of the particles.
   * @param forces Iterator containing the forces acting on each particle.
   * @param numberParticles The length of the input iterators.
   * @param st CUDA stream to use for asynchronous operations.
   * @tparam PosIterator Type of the position iterator.
   * @tparam ForceIterator Type of the force iterator.
   * @return A cached_vector containing the computed hydrodynamic displacements
   (velocities) for each particle.
   */
  template <class PosIterator, class ForceIterator>
  cached_vector<real3> Mdot(PosIterator pos, ForceIterator forces,
                            int numberParticles, cudaStream_t st = 0) {
    auto M = Mdot(pos, forces, (real4 *)nullptr, numberParticles, st);
    return M.first;
  }

  /**
   * @brief Computes the linear and angular hydrodynamic displacements
   * (velocities) coming from the forces and torques acting on a group of
   * positions. If the torques pointer is null, the function will only compute
   * and return the translational part of the mobility.
   * @param pos Iterator containing the positions of the particles.
   * @param forces Iterator containing the forces acting on each particle.
   * @param torques Iterator containing the torques acting on each particle.
   * @param numberParticles The length of the input iterators.
   * @param st CUDA stream to use for asynchronous operations.
   * @tparam PosIterator Type of the position iterator.
   * @tparam ForceIterator Type of the force iterator.
   * @tparam TorqueIterator Type of the torque iterator.
   * @return A pair of cached_vectors containing the computed hydrodynamic
   * displacements (velocities) for each particle, and the angular velocities if
   * torques are provided.
   */
  template <class PosIterator, class ForceIterator, class TorqueIterator>
  std::pair<cached_vector<real3>, cached_vector<real3>>
  Mdot(PosIterator pos, ForceIterator forces, TorqueIterator torques,
       int numberParticles, cudaStream_t st) {
    cudaDeviceSynchronize();
    System::log<System::DEBUG2>("[DPStokes] Computing displacements");
    auto gridData = ibm->spreadForces(pos, forces, numberParticles, st);
    auto gridForceCheb = fct->forwardTransform(gridData, st);
    if (torques) { // Torques are added in Cheb space
      ibm->addSpreadTorquesFourier(pos, torques, numberParticles, gridForceCheb,
                                   fct, st);
    }
    FluidData<complex> fluid = solveBVPVelocity(gridForceCheb, st);
    if (mode != WallMode::none) {
      correction->correctSolution(fluid, gridForceCheb, st);
    }
    cached_vector<real3> particleAngularVelocities;
    if (torques) {
      // Ang. velocities are interpolated from the curl of the velocity, which
      // is
      //  computed in Cheb space.
      auto gridAngVelsCheb = ibm->computeGridAngularVelocityCheb(fluid, st);
      auto gridAngVels = fct->inverseTransform(gridAngVelsCheb, st);
      particleAngularVelocities = ibm->interpolateAngularVelocity(
          gridAngVels, pos, numberParticles, st);
    }
    gridData = fct->inverseTransform(fluid.velocity, st);
    auto particleVelocities =
        ibm->interpolateVelocity(gridData, pos, numberParticles, st);
    CudaCheckError();
    return {particleVelocities, particleAngularVelocities};
  }

  /**
   * @brief Computes the average velocity profile in the x or y direction as a
   * function of z.
   *
   * This function calculates the average velocity of particles along the
   * specified direction (x or y) as a function of the z-coordinate. It does so
   * by spreading the particle forces onto a grid, performing a Chebyshev
   * transform, solving the boundary value problem for the fluid velocity,
   * applying corrections if necessary, and then reconstructing the average
   * velocity profile in real space using Chebyshev polynomial summation.
   *
   * @tparam PosIterator Iterator type for particle positions.
   * @tparam ForceIterator Iterator type for particle forces.
   * @param pos Iterator to the beginning of particle positions.
   * @param forces Iterator to the beginning of particle forces.
   * @param numberParticles Number of particles in the system.
   * @param direction Direction in which to compute the average velocity: 0 for
   * x (default), 1 for y.
   * @param st CUDA stream to use for asynchronous operations (default: 0).
   * @return std::vector<double> Average velocity as a function of z (size:
   * number of grid points in z).
   */
  template <class PosIterator, class ForceIterator>
  std::vector<double>
  computeAverageVelocity(PosIterator pos, ForceIterator forces,
                         int numberParticles, int direction = 0,
                         cudaStream_t st = 0) {
    cudaDeviceSynchronize();
    System::log<System::DEBUG2>("[DPStokes] Computing displacements");
    auto gridData = ibm->spreadForces(pos, forces, numberParticles, st);
    auto gridForceCheb = fct->forwardTransform(gridData, st);
    FluidData<complex> fluid = solveBVPVelocity(gridForceCheb, st);
    if (mode != WallMode::none) {
      correction->correctSolution(fluid, gridForceCheb, st);
    }
    int3 n = this->grid.cellDim;
    std::vector<complex> chebCoeff;
    if (direction == 0)
      chebCoeff = detail::getZeroModeChebCoeff(fluid.velocity.m_x, n);
    else if (direction == 1)
      chebCoeff = detail::getZeroModeChebCoeff(fluid.velocity.m_y, n);
    else
      throw std::runtime_error(
          "[DPStokesSlab] Can only average in direction X (0) or Y (1)");

    std::vector<double> averageVelocity(n.z);
    // transfer to real space by direct summation
    // Chebyshev stuff refresher
    // f(z) = c_0+c_1T_1(z)+c_2T_2(z)+...
    // z = (b+a)/2+(b-a)/2*cos(i*M_PI/(nz-1));
    // arg = acos(-1+2*(z-a)/(b-a));
    // T_j(z) = cos(j acos(-1+2*(z-a)/(b-a))) with z =
    // (b+a)/2+(b-a)/2*cos(i*M_PI/(nz-1))
    for (int i = 0; i < n.z; i++) {
      real arg = i * M_PI / (n.z - 1);
      for (int j = 0; j < n.z; j++) {
        averageVelocity[i] += chebCoeff[j].x * cos(j * arg);
      }
    }

    CudaCheckError();
    return averageVelocity;
  }

private:
  shared_ptr<FastChebyshevTransform> fct;
  shared_ptr<Correction> correction;
  shared_ptr<SpreadInterp> ibm;
  gpu_container<real> zeroModeVelocityChebyshevIntegrals;
  gpu_container<real> zeroModePressureChebyshevIntegrals;

  void setUpGrid(Parameters par);
  void initializeKernel(Parameters par);
  void printStartingMessages(Parameters par);
  void resizeVectors();
  void initializeBoundaryValueProblemSolver();

  void precomputeIntegrals();
  void resetGridForce();
  void tryToResetGridForce();
  FluidData<complex> solveBVPVelocity(DataXYZ<complex> &gridForcesFourier,
                                      cudaStream_t st);
  void resizeTmpStorage(size_t size);
  real Lx, Ly;
  real H;
  Grid grid;
  real viscosity;
  real gw;
  real tolerance;
  WallMode mode;
  shared_ptr<BVP::BatchedBVPHandlerReal> bvpSolver;
};

namespace detail {
struct LanczosAdaptor : lanczos::MatrixDot {
  std::shared_ptr<DPStokes> dpstokes;
  real4 *pos;
  int numberParticles;

  LanczosAdaptor(std::shared_ptr<DPStokes> dpstokes, real4 *pos,
                 int numberParticles)
      : dpstokes(dpstokes), pos(pos), numberParticles(numberParticles) {}

  void operator()(real *v, real *mv) override {
    auto res = dpstokes->Mdot(pos, (real3 *)v, numberParticles);
    thrust::copy(thrust::cuda::par, res.begin(), res.begin() + numberParticles,
                 (real3 *)mv);
  }
};

struct SaruTransform {
  uint s1, s2;
  real std;
  SaruTransform(uint s1, uint s2, real std = 1.0) : s1(s1), s2(s2), std(std) {}

  __device__ real3 operator()(uint id) {
    Saru rng(s1, s2, id);
    return make_real3(rng.gf(0.0f, std), rng.gf(0.0f, std).x);
  }
};

auto fillRandomVectorReal3(int numberParticles, uint s1, uint s2,
                           real std = 1.0) {
  cached_vector<real3> noise(numberParticles);
  auto cit = thrust::make_counting_iterator<uint>(0);
  auto tr = thrust::make_transform_iterator(cit, SaruTransform(s1, s2, std));
  thrust::copy(thrust::cuda::par, tr, tr + numberParticles, noise.begin());
  return noise;
}

struct SumPosAndNoise {
  real3 *b;
  real4 *pos;
  real sign;
  SumPosAndNoise(real4 *pos, real3 *b, real sign)
      : pos(pos), b(b), sign(sign) {}

  __device__ auto operator()(int id) {
    return make_real3(pos[id]) + sign * b[id];
  }
};

struct BDIntegrate {
  real4 *pos;
  real3 *mf;
  real3 *noise;
  real3 *noisePrev;
  real3 *rfd;
  real dt;
  BDIntegrate(real4 *pos, real3 *mf, real3 *noise, real3 *noisePrev, real3 *rfd,
              real dt, real temperature)
      : pos(pos), mf(mf), noise(noise), noisePrev(noisePrev), rfd(rfd), dt(dt) {
  }
  __device__ void operator()(int id) {
    real3 displacement = mf[id] * dt;
    if (noise) {
      real3 fluct;
      if (noisePrev)
        fluct = dt * real(0.5) * (noise[id] + noisePrev[id]);
      else
        fluct = dt * noise[id];
      displacement += fluct + rfd[id];
    }
    pos[id] += make_real4(displacement);
  }
};

} // namespace detail
/**
 * @brief A wrapper class exposing DPStokes as an Integrator
 */
class DPStokesIntegrator : public Integrator {
  int steps = 0;
  uint seed, seedRFD;
  std::shared_ptr<DPStokes> dpstokes;
  std::shared_ptr<lanczos::Solver> lanczos;
  thrust::device_vector<real3> previousNoise;
  real deltaRFD;
  uint stepsRFD =
      0; // How many times we have updated the RFD (for random numbers)
  uint stepsLanczos =
      0; // How many times we have updated Lanczos (for random numbers)
public:
  template <class T> using cached_vector = cached_vector<T>;
  /**
   * @brief Parameters for the DPStokesIntegrator. This class needs all the
   * DPStokes parameters, plus the ones in this struct.
   */
  struct Parameters : DPStokes::Parameters {
    real temperature = 0;       ///< Temperature in units of \f$k_B T\f$
    bool useLeimkuhler = false; ///< Use Leimkuhler thermostat, will use
                                ///< Euler-maruyama otherwise. (default: false)
    real dt;                    ///< Time step
    real tolerance = 1e-7;      ///< Tolerance for the Lanczos algorithm
  };

  /**
   * @brief Construct a new DPStokesIntegrator
   *
   * @param pd Particle data instance
   * @param par Parameters for the integrator
   */
  DPStokesIntegrator(std::shared_ptr<ParticleData> pd, Parameters par)
      : Integrator(pd, "DPStokes"), par(par) {
    dpstokes = std::make_shared<DPStokes>(par);
    lanczos = std::make_shared<lanczos::Solver>();
    System::log<System::MESSAGE>("[DPStokes] dt %g", par.dt);
    System::log<System::MESSAGE>("[DPStokes] temperature: %g", par.temperature);
    this->seed = pd->getSystem()->rng().next32();
    this->seedRFD = pd->getSystem()->rng().next32();
    this->deltaRFD = 1e-6;
  }

  /**
   * @brief Advance the simulation by one time step.
   */
  void forwardTime() {
    System::log<System::DEBUG2>("[DPStokes] Running step %d", steps);
    if (steps == 0)
      setUpInteractors();
    resetForces();
    for (auto i : interactors) {
      i->updateSimulationTime(par.dt * steps);
      i->sum({.force = true});
    }
    const int numberParticles = pd->getNumParticles();
    auto mf = computeDeterministicDisplacements();
    if (par.temperature) {
      auto bdw = computeFluctuations();
      if (par.useLeimkuhler and previousNoise.size() != numberParticles) {
        previousNoise.resize(numberParticles);
        thrust::copy(bdw.begin(), bdw.end(), previousNoise.begin());
      }
      auto rfd = computeThermalDrift();
      real3 *d_prevNoise =
          par.useLeimkuhler ? previousNoise.data().get() : nullptr;
      auto pos = pd->getPos(access::gpu, access::readwrite);
      detail::BDIntegrate bd(pos.raw(), mf.data().get(), bdw.data().get(),
                             d_prevNoise, rfd.data().get(), par.dt,
                             par.temperature);
      auto cit = thrust::make_counting_iterator(0);
      thrust::for_each_n(thrust::cuda::par, cit, numberParticles, bd);
      if (par.useLeimkuhler)
        thrust::copy(bdw.begin(), bdw.end(), previousNoise.begin());
    } else {
      auto pos = pd->getPos(access::gpu, access::readwrite);
      detail::BDIntegrate bd(pos.raw(), mf.data().get(), nullptr, nullptr,
                             nullptr, par.dt, 0);
      auto cit = thrust::make_counting_iterator(0);
      thrust::for_each_n(thrust::cuda::par, cit, numberParticles, bd);
    }
    steps++;
  }

private:
  Parameters par;

  void setUpInteractors() {
    Box box({par.Lx, par.Ly, par.H});
    box.setPeriodicity(1, 1, 0);
    for (auto i : interactors) {
      i->updateBox(box);
      i->updateTimeStep(par.dt);
      i->updateTemperature(par.temperature);
      i->updateViscosity(par.viscosity);
    }
  }
  void resetForces() {
    auto force = pd->getForce(access::gpu, access::write);
    thrust::fill(thrust::cuda::par, force.begin(), force.end(), real4());
  }
  // Returns the thermal drift term: temperature*dt*(\partial_q \cdot M)
  auto computeThermalDrift() {
    if (par.temperature) {
      auto pos = pd->getPos(access::gpu, access::read);
      const int numberParticles = pd->getNumParticles();
      this->stepsRFD++;
      auto noise2 =
          detail::fillRandomVectorReal3(numberParticles, seedRFD, stepsRFD);
      auto cit = thrust::make_counting_iterator(0);
      auto posp = thrust::make_transform_iterator(
          cit, detail::SumPosAndNoise(pos.raw(), noise2.data().get(),
                                      deltaRFD * 0.5));
      auto mpw = dpstokes->Mdot(posp, noise2.data().get(), numberParticles, 0);
      auto posm = thrust::make_transform_iterator(
          cit, detail::SumPosAndNoise(pos.raw(), noise2.data().get(),
                                      -deltaRFD * 0.5));
      auto mmw = dpstokes->Mdot(posm, noise2.data().get(), numberParticles, 0);
      using namespace thrust::placeholders;
      thrust::transform(mpw.begin(), mpw.end(), mmw.begin(), mpw.begin(),
                        make_real3(par.dt * par.temperature / deltaRFD) *
                            (_1 - _2));
      return mpw;
    } else {
      return cached_vector<real3>();
    }
  }

  // Returns sqrt(2*M*temperature/dt)dW
  auto computeFluctuations() {
    const int numberParticles = pd->getNumParticles();
    cached_vector<real3> bdw(numberParticles);
    thrust::fill(bdw.begin(), bdw.end(), real3());
    if (par.temperature) {
      this->stepsLanczos++;
      auto pos = pd->getPos(access::gpu, access::read);
      detail::LanczosAdaptor dot(dpstokes, pos.raw(), numberParticles);
      auto noise =
          detail::fillRandomVectorReal3(numberParticles, seed, stepsLanczos,
                                        sqrt(2 * par.temperature / par.dt));
      lanczos->run(dot, (real *)bdw.data().get(), (real *)noise.data().get(),
                   par.tolerance, 3 * numberParticles);
    }
    return bdw;
  }

  // Returns the product of the forces and the mobility matrix, M F
  auto computeDeterministicDisplacements() {
    auto pos = pd->getPos(access::gpu, access::read);
    auto force = pd->getForce(access::gpu, access::read);
    if (pd->isTorqueAllocated()) {
      System::log<System::EXCEPTION>(
          "[DPStokes] Torques are not yet implemented");
      throw std::runtime_error("Operation not implemented");
    }
    const int numberParticles = pd->getNumParticles();
    return dpstokes->Mdot(pos.raw(), force.raw(), numberParticles, 0);
  }
};
} // namespace DPStokesSlab_ns
} // namespace uammd
#include "StokesSlab/DPStokes.cu"
#include "StokesSlab/initialization.cu"
#endif

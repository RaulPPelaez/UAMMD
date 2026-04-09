/*Raul P. Pelaez 2020-2021. Force Biased Monte Carlo Integrator. The algorithm
encoded in this module is the so called Metropolized Forward Euler-Maruyama in
[1]. Also known as Metropolis-adjusted Langevin algorithm (MALA).


USAGE:
Use as any other integrator module

...
MC::ForceBiased::Parameters par;
par.beta = 1.0; //Inverse of temperature
par.stepSize = 0.0001; //Initial step size (will be auto optimized)
par.acceptanceRatio = 0.5;//Desired ratio of trial acceptance/rejection.
auto mala = std::make_shared<MC::ForceBiased>(pd, par);
...
mala->addInteractor(myinteractor);
...
//Try new configurations until one is accepted
mala->forwardTime();

//Get current energy
real currentEnergy = mala->getCurrentEnergy();
//Equivalent to
  real currentEnergy = 0;
  {
    auto energy = pd->getEnergy(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par, energy.begin(), energy.end(), real());
  }
  auto interactors = mala->getInteractors();
  for(auto &i: interactors){
    currentEnergy += i->sumEnergy();
  }
  auto energy = pd->getEnergy(access::location::gpu, access::mode::read);
  currentEnergy += thrust::reduce(thrust::cuda::par, energy.begin(),
energy.end());

//Get current step size and acceptance ratio
real currentStepSize = mala->getCurrentStepSize();
real currentAcceptanceRatio = mala->getCurrentAcceptanceRatio();


REFERENCES:
[1] Pathwise Accuracy and Ergodicity of Metropolized Integrators for SDEs
(2009). NAWAF BOU-RABEE and ERIC VANDEN-EIJNDEN.
https://doi.org/10.1002/cpa.20306
 */
#include "Integrator/Integrator.cuh"
#include "third_party/saruprng.cuh"
#include "utils/debugTools.h"
#include "utils/exception.h"

namespace uammd {
namespace MC {
namespace forcebiased_ns {

struct EulerMaruyama {
  real h, noiseAmp;
  uint seed, step;
  EulerMaruyama(uint seed, uint step, real h, real beta)
      : seed(seed), step(step), h(h) {
    noiseAmp = sqrt(2.0 * h / beta);
  }

  __device__ real4 operator()(real4 pos,
                              thrust::tuple<real4, int> forceAndIndex) {
    int i = thrust::get<1>(forceAndIndex);
    real3 force = make_real3(thrust::get<0>(forceAndIndex));
    Saru saru(seed, step, i);
    real3 noise = make_real3(saru.gf(0, 1), saru.gf(0, 1).x);
    return pos + make_real4(force * h + noiseAmp * noise);
  }
};

// From eq. 2.4 in [1]
struct TransitionKernel {
  real h;
  TransitionKernel(real h) : h(h) {}

  __device__ real operator()(thrust::tuple<real4, real4, real4> XYForceX) {
    real3 X = make_real3(thrust::get<0>(XYForceX));
    real3 Y = make_real3(thrust::get<1>(XYForceX));
    real3 ForceX = make_real3(thrust::get<2>(XYForceX));
    real3 element = Y - X - h * ForceX;
    return dot(element, element);
  }
};

class Optimize {
  int naccept;
  int ntry;
  int ncontrol;
  real jumpSize;
  real currentAcceptanceRatio = 0;
  const real targetRatio = 0.9;

  void adjustStepSize() {
    constexpr float updateRateIncrease = 1.02;
    constexpr float updateRateDecrease = 0.9;
    constexpr float minimumJumpSize = 1e-8;
    constexpr float maximumJumpSize = 2;
    const float ratio = naccept / (float)ntry;
    if (ratio > targetRatio and jumpSize < maximumJumpSize) {
      jumpSize *= updateRateIncrease;
    } else if (ratio < targetRatio and jumpSize > minimumJumpSize) {
      jumpSize *= updateRateDecrease;
    }
    ntry = naccept = 0;
    currentAcceptanceRatio = ratio;
  }

public:
  Optimize(real targetAcceptanceRatio, real initialStepSize = 1)
      : targetRatio(targetAcceptanceRatio) {
    naccept = 0;
    ntry = 0;
    ncontrol = 1000;
    jumpSize = initialStepSize;
  }

  void registerAccept() {
    naccept++;
    ntry++;
    if (ntry % ncontrol == 0) {
      adjustStepSize();
    }
  }

  void registerReject() {
    ntry++;
    if (ntry % ncontrol == 0) {
      adjustStepSize();
    }
  }

  real getStepSize() { return jumpSize; }

  real getCurrentAcceptanceRatio() { return currentAcceptanceRatio; }
};

} // namespace forcebiased_ns

class ForceBiased : public Integrator {
  template <class T> using gpu_container = thrust::device_vector<T>;
  cudaStream_t st;
  gpu_container<real4> storedForce, storedPos;
  real currentEnergy, storedEnergy;
  real beta;
  uint step = 0;
  uint seed;
  forcebiased_ns::Optimize optimizeStepSize;

public:
  struct Parameters {
    real beta = -1;      // Inverse of temperature
    real stepSize = 0.1; // Initial step length (will be optimized according to
                         // the target acceptance ratio)
    real acceptanceRatio = 0.5; // Desired acceptance ratio
  };

  ForceBiased(shared_ptr<ParticleData> pd, Parameters par)
      : ForceBiased(std::make_shared<ParticleGroup>(pd, "All"), par) {}

  ForceBiased(shared_ptr<ParticleGroup> pg, Parameters par)
      : Integrator(pg, "MC::ForceBiased"), beta(par.beta),
        optimizeStepSize(par.acceptanceRatio, par.stepSize) {
    sys->log<System::MESSAGE>("[MC::ForceBiased] Initialized");
    sys->log<System::MESSAGE>("[MC::ForceBiased] Temperature: %g", 1.0 / beta);
    sys->log<System::MESSAGE>("[MC::ForceBiased] Target acceptance ratio: %g",
                              par.acceptanceRatio);
    sys->log<System::MESSAGE>("[MC::ForceBiased] Initial step size: %g",
                              par.stepSize);
    if (beta < 0 or par.stepSize <= 0 or par.acceptanceRatio <= 0) {
      sys->log<System::ERROR>("[MC::ForceBiased] ERROR: parameters beta, "
                              "stepSize and acceptanceRatio must be >=0");
      throw std::runtime_error("Invalid parameter");
    }
    CudaSafeCall(cudaStreamCreate(&st));
    this->seed = sys->rng().next32();
  }

  ~ForceBiased() { cudaStreamDestroy(st); }

  real getCurrentEnergy() { return currentEnergy; }

  real getCurrentStepSize() { return optimizeStepSize.getStepSize(); }

  real getCurrentAcceptanceRatio() {
    return optimizeStepSize.getCurrentAcceptanceRatio();
  }

  void forwardTime() {
    sys->log<System::DEBUG1>("[ForceBiased] Starting step %d", step);
    if (step == 0) {
      firstStep();
    }
    updateInteractors();
    while (!tryNewStep()) {
    }
    CudaCheckError();
  }

private:
  void updateInteractors() {
    if (step == 0) {
      for (auto updatable : updatables) {
        updatable->updateTemperature(1.0 / beta);
      }
    }
  }

  void firstStep() {
    sys->log<System::DEBUG1>("[ForceBiased] First step");
    storePositions();
    updateForceEnergyEstimation();
    storeForces();
    storeEnergy();
    CudaCheckError();
  }

  bool tryNewStep() {
    step++;
    proposeNewStep();
    if (isNewConfigurationAccepted()) {
      storeCurrentConfiguration();
      optimizeStepSize.registerAccept();
      return true;
    } else {
      restoreStoredConfiguration();
      optimizeStepSize.registerReject();
      return false;
    }
    CudaCheckError();
  }

  bool isNewConfigurationAccepted() {
    const real internalEnergyX = storedEnergy;
    real transitionKernelXY;
    {
      auto currentPos = pd->getPos(access::location::gpu, access::mode::read);
      transitionKernelXY =
          computeTransitionKernel(storedPos, currentPos, storedForce);
    }
    const real internalEnergyY = updateForceEnergyEstimation();
    real transitionKernelYX;
    {
      auto currentPos = pd->getPos(access::location::gpu, access::mode::read);
      auto currentForce =
          pd->getForce(access::location::gpu, access::mode::read);
      transitionKernelYX =
          computeTransitionKernel(currentPos, storedPos, currentForce);
    }
    const real metropolisExponent = internalEnergyY - internalEnergyX;
    const real transitionProbabilityDensityExponent =
        transitionKernelYX - transitionKernelXY;
    // Acceptance probability from eq. 2.7 in [1]
    real acceptanceProbability = exp(
        -beta * (metropolisExponent + transitionProbabilityDensityExponent));
    const real Z = sys->rng().uniform(0, 1);
    const bool acceptNewConfiguration = Z < acceptanceProbability;
    CudaCheckError();
    return acceptNewConfiguration;
  }

  void storeCurrentConfiguration() {
    storePositions();
    storeForces();
    storeEnergy();
  }

  void restoreStoredConfiguration() {
    auto pos = pd->getPos(access::location::gpu, access::mode::write);
    thrust::copy(thrust::cuda::par.on(st), storedPos.begin(), storedPos.end(),
                 pos.begin());
    auto force = pd->getForce(access::location::gpu, access::mode::write);
    thrust::copy(thrust::cuda::par.on(st), storedForce.begin(),
                 storedForce.end(), force.begin());
    currentEnergy = storedEnergy;
  }

  void storePositions() {
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    storedPos.resize(pg->getNumberParticles());
    thrust::copy(thrust::cuda::par.on(st), pos.begin(), pos.end(),
                 storedPos.begin());
  }

  void storeForces() {
    auto force = pd->getForce(access::location::gpu, access::mode::read);
    storedForce.resize(pg->getNumberParticles());
    thrust::copy(thrust::cuda::par.on(st), force.begin(), force.end(),
                 storedForce.begin());
  }

  void storeEnergy() { storedEnergy = getCurrentEnergy(); }

  void proposeNewStep() {
    sys->log<System::DEBUG2>("[ForceBiased] Propose next move");
    real stepSize = optimizeStepSize.getStepSize();
    auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
    auto force = pd->getForce(access::location::gpu, access::mode::read);
    auto cit = thrust::make_counting_iterator<int>(0);
    auto forceAndIndex =
        thrust::make_zip_iterator(thrust::make_tuple(force.begin(), cit));
    thrust::transform(
        thrust::cuda::par.on(st), pos.begin(), pos.end(), forceAndIndex,
        pos.begin(), forcebiased_ns::EulerMaruyama(seed, step, stepSize, beta));
    CudaCheckError();
  }

  // From eq. 2.4 in [1]
  template <class Container1, class Container2, class Container3>
  real computeTransitionKernel(Container1 &posX, Container2 &posY,
                               Container3 &gradUX) {
    real stepSize = optimizeStepSize.getStepSize();
    int numberElements = std::distance(posX.begin(), posX.end());
    auto zip = thrust::make_zip_iterator(
        thrust::make_tuple(posX.begin(), posY.begin(), gradUX.begin()));
    auto expDenominator = thrust::transform_reduce(
        thrust::cuda::par.on(st), zip, zip + numberElements,
        forcebiased_ns::TransitionKernel(stepSize), real(),
        cuda::std::plus<real>());
    CudaCheckError();
    return expDenominator / (4.0 * stepSize);
  }

  void resetForceAndEnergy() {
    auto force = pd->getForce(access::location::gpu, access::mode::write);
    auto energy = pd->getEnergy(access::location::gpu, access::mode::write);
    thrust::fill(thrust::cuda::par.on(st), force.begin(), force.end(), real4());
    thrust::fill(thrust::cuda::par.on(st), energy.begin(), energy.end(),
                 real());
    currentEnergy = 0;
    CudaCheckError();
  }

  real updateForceEnergyEstimation() {
    sys->log<System::DEBUG2>(
        "[MC::ForceBiased] Computing current force and energy");
    resetForceAndEnergy();
    for (auto inter : interactors) {
      inter->sum({.force = true, .energy = true, .virial = false}, st);
    }
    auto energy = pd->getEnergy(access::location::gpu, access::mode::read);
    currentEnergy +=
        thrust::reduce(thrust::cuda::par.on(st), energy.begin(), energy.end());
    currentEnergy *= 0.5;
    if (isnan(currentEnergy) or isinf(currentEnergy)) {
      currentEnergy = std::numeric_limits<real>::max();
      sys->log<System::WARNING>(
          "[MC::ForceBiased] Detected an infinite or NaN energy");
    }
    CudaCheckError();
    return currentEnergy;
  }
};
} // namespace MC

} // namespace uammd

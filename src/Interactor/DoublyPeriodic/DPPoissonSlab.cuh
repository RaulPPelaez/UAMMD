/*Raul P. Pelaez 2019-2020. Spectral/Chebyshev Doubly Periodic Poisson solver.
   Slab geometry

 */

#ifndef DOUBLYPERIODIC_POISSON_SLAB_CUH
#define DOUBLYPERIODIC_POISSON_SLAB_CUH

#include "Interactor/DoublyPeriodic/PoissonSlab/FarField.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/NearField.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include "Interactor/Interactor.cuh"
#include "utils/utils.h"
#include <thrust/device_vector.h>
namespace uammd {

class DPPoissonSlab : public Interactor {
public:
  using Permitivity = DPPoissonSlab_ns::Permitivity;
  using SurfaceValueDispatch = DPPoissonSlab_ns::SurfaceValueDispatch;
  struct Parameters {
    real upsampling = 1.2;
    real2 Lxy;
    int Nxy = -1;
    real H;
    Permitivity permitivity;
    real tolerance = 1e-4;
    real gw = -1;
    int support = 12;
    real numberStandardDeviations = 4;
    real split = -1;
    // Either Charge of potential depending on whether the walls are metallic or
    // not
    std::shared_ptr<SurfaceValueDispatch> surfaceValues =
        std::make_shared<SurfaceValueDispatch>();

    bool printK0Mode = false;
  };

  DPPoissonSlab(shared_ptr<ParticleGroup> pg, Parameters par);

  DPPoissonSlab(shared_ptr<ParticleData> pd, Parameters par)
      : DPPoissonSlab(std::make_shared<ParticleGroup>(pd, "All"), par) {}

  ~DPPoissonSlab() { sys->log<System::MESSAGE>("[DPPoissonSlab] Destroyed"); }

  void sum(Computables comp, cudaStream_t st = 0) override {
    farField->compute(st);
    nearField->compute(st);
    if (comp.virial) {
      sys->log<System::EXCEPTION>(
          "[Poisson] Virial functionality not implemented.");
      throw std::runtime_error("[Poisson] not implemented");
    }
  }

  thrust::device_vector<real4> computeFieldAtParticles() {
    thrust::device_vector<real4> fieldAtParticles(pg->getNumberParticles());
    thrust::fill(fieldAtParticles.begin(), fieldAtParticles.end(), real4());
    auto field_ptr = thrust::raw_pointer_cast(fieldAtParticles.data());
    farField->compute(0, field_ptr);
    nearField->compute(0, field_ptr);
    return fieldAtParticles;
  }

  void setSurfaceValuesZeroModeFourier(cufftComplex2_t<real> zeroMode) {
    farField->setSurfaceValuesZeroModeFourier(zeroMode);
  }

  auto getK0Mode() { return this->farField->getK0Mode(); }

private:
  shared_ptr<DPPoissonSlab_ns::NearField> nearField;
  shared_ptr<DPPoissonSlab_ns::FarField> farField;

  void initializeNearField(Parameters par);
  void initializeFarField(Parameters par);
  void printStartingMessages(Parameters par);
};

} // namespace uammd

#include "Interactor/DoublyPeriodic/PoissonSlab/initialization.cu"
#endif

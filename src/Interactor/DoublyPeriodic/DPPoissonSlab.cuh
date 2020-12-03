/*Raul P. Pelaez 2019-2020. Spectral/Chebyshev Doubly Periodic Poisson solver. Slab geometry



 */

#ifndef DOUBLYPERIODIC_POISSON_SLAB_CUH
#define DOUBLYPERIODIC_POISSON_SLAB_CUH

#include "Interactor/Interactor.cuh"
#include "utils/utils.h"
#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/NearField.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/FarField.cuh"
namespace uammd{

  class DPPoissonSlab: public Interactor{
  public:
    using Permitivity = DPPoissonSlab_ns::Permitivity;
    using SurfaceChargeDispatch = DPPoissonSlab_ns::SurfaceChargeDispatch;
    struct Parameters{
      real upsampling = -1.0;
      int3 cells = make_int3(-1, -1, -1); //Number of Fourier nodes in each direction
      real2 Lxy;
      real H;
      Permitivity permitivity;
      real tolerance = 1e-4;
      real gw = -1;
      int support = -1;
      real numberStandardDeviations = -1;
      real split = -1;
      std::shared_ptr<SurfaceChargeDispatch> surfaceCharge = std::make_shared<SurfaceChargeDispatch>();
    };

    DPPoissonSlab(shared_ptr<ParticleData> pd,
		  shared_ptr<ParticleGroup> pg,
		  shared_ptr<System> sys,
		  Parameters par);
    ~DPPoissonSlab(){
      sys->log<System::MESSAGE>("[DPPoissonSlab] Destroyed");
    }

    void sumForce(cudaStream_t st){
      sys->log<System::DEBUG2>("[DPPoissonSlab] Sum Force");
      sumForceEnergy(st);
    }

    real sumEnergy(){
      sys->log<System::DEBUG2>("[DPPoissonSlab] Sum Energy");
      return sumForceEnergy(0);
    }
    
    real sumForceEnergy(cudaStream_t st){
      farField->compute(st);
      nearField->compute(st);
      return 0;
    }
    
  private:
    shared_ptr<DPPoissonSlab_ns::NearField> nearField;
    shared_ptr<DPPoissonSlab_ns::FarField> farField;

    void initializeNearField(Parameters par);
    void initializeFarField(Parameters par);
    void printStartingMessages(Parameters par);

    real numberStandardDeviations;
  };

}

#include"Interactor/DoublyPeriodic/PoissonSlab/initialization.cu"
#endif


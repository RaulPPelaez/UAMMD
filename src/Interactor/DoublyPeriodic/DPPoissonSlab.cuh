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
    using SurfaceValueDispatch = DPPoissonSlab_ns::SurfaceValueDispatch;
    struct Parameters{
      real upsampling = 1.2;
      real2 Lxy;
      int Nxy = -1;
      real H;
      Permitivity permitivity;
      real tolerance = 1e-4;
      real gw = -1;
      int support = 10;
      real numberStandardDeviations = 4;
      real split = -1;
      //Either Charge of potential depending on whether the walls are metallic or not
      std::shared_ptr<SurfaceValueDispatch> surfaceValues = std::make_shared<SurfaceValueDispatch>();
    };

    DPPoissonSlab(shared_ptr<ParticleGroup> pg, Parameters par);

    DPPoissonSlab(shared_ptr<ParticleData> pd, Parameters par):
      DPPoissonSlab(std::make_shared<ParticleGroup>(pd, "All"), par){}

    ~DPPoissonSlab(){
      sys->log<System::MESSAGE>("[DPPoissonSlab] Destroyed");
    }
    
    virtual void sum(Computables comp, cudaStream_t st = 0) override{
      farField->compute(st);
      nearField->compute(st);
      if(comp.virial){
	sys->log<System::EXCEPTION>("[Poisson] Virial functionality not implemented.");
	throw std::runtime_error("[Poisson] not implemented");
      }
    }

    void setSurfaceValuesZeroModeFourier(cufftComplex2_t<real> zeroMode){
      farField->setSurfaceValuesZeroModeFourier(zeroMode);
    }
    
  private:
    shared_ptr<DPPoissonSlab_ns::NearField> nearField;
    shared_ptr<DPPoissonSlab_ns::FarField> farField;

    void initializeNearField(Parameters par);
    void initializeFarField(Parameters par);
    void printStartingMessages(Parameters par);
  };

}

#include"Interactor/DoublyPeriodic/PoissonSlab/initialization.cu"
#endif


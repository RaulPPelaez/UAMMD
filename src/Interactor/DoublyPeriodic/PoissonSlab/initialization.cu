/*Raul P. Pelaez 2020. Spectral/Chebyshev Doubly Periodic Poisson Slab solver.
 * Initialization functions
 */
#include"Interactor/DoublyPeriodic/DPPoissonSlab.cuh"
#include"Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include "misc/ChevyshevUtils.cuh"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "utils.cuh"
namespace uammd{

  namespace detail{
    real computeNumberStandardDeviations(real tolerance){
      //TODO: This number should change with tolerance
      return 4;
    }
    
    void throwIfInvalidSplit(real numberStandardDeviations, DPPoissonSlab::Parameters par){
      constexpr real upperRangeFactor = 3.2; //TODO: Hardcoded for 1e-3 tolerance, should change
      real minimumSplit = upperRangeFactor/(par.H + numberStandardDeviations*par.gw);
      real maximumSplit = 1/(par.gw*sqrt(12));
      if(par.split < minimumSplit or par.split > maximumSplit){
	System::log<System::ERROR>("Splitting parameter is outside of valid range [%g:%g] (got %g)",
				   minimumSplit, maximumSplit, par.split);
	throw std::invalid_argument("Invalid splitting parameter");
      }
    }
  }

  DPPoissonSlab::DPPoissonSlab(shared_ptr<ParticleData> pd,
		       shared_ptr<ParticleGroup> pg,
		       shared_ptr<System> sys,
		       DPPoissonSlab::Parameters par):
    Interactor(pd, pg, sys, "IBM::DPPoissonSlab"){
    this->numberStandardDeviations = par.numberStandardDeviations;
    if(this->numberStandardDeviations<=0){
      this->numberStandardDeviations = detail::computeNumberStandardDeviations(par.tolerance);
    }
    detail::throwIfInvalidSplit(numberStandardDeviations, par);
    printStartingMessages(par);
    initializeFarField(par);
    initializeNearField(par);
    CudaCheckError();
  }

  void DPPoissonSlab::printStartingMessages(Parameters par){
    sys->log<System::MESSAGE>("[DPPoissonSlab] tolerance: %g", par.tolerance);
    sys->log<System::MESSAGE>("[DPPoissonSlab] permitivity: %g (inside), %g (top), %g (bottom)",
			      par.permitivity.inside, par.permitivity.top, par.permitivity.bottom);
    sys->log<System::MESSAGE>("[DPPoissonSlab] Gaussian source width: %g", par.gw);
    sys->log<System::WARNING>("[DPPoissonSlab] Not subtracting adding phi(0,0,0)");
  }

  void DPPoissonSlab::initializeNearField(Parameters par){
    DPPoissonSlab_ns::NearField::Parameters nfpar;
    nfpar.split = par.split;
    nfpar.permitivity = par.permitivity;
    nfpar.gw = par.gw;
    nfpar.H = par.H;
    nfpar.Lxy = par.Lxy;
    nfpar.tolerance = par.tolerance;
    nfpar.numberStandardDeviations = numberStandardDeviations;
    this->nearField = std::make_shared<DPPoissonSlab_ns::NearField>(sys, pd, pg, nfpar);
  }

  void DPPoissonSlab::initializeFarField(Parameters par){
    DPPoissonSlab_ns::FarField::Parameters ffpar;
    ffpar.split = par.split;
    ffpar.permitivity = par.permitivity;
    ffpar.gw = par.gw;
    ffpar.H = par.H;
    ffpar.numberStandardDeviations = numberStandardDeviations;
    ffpar.Lxy = par.Lxy;
    ffpar.tolerance = par.tolerance;
    ffpar.upsampling = par.upsampling;
    ffpar.surfaceCharge = par.surfaceCharge;
    ffpar.cells = par.cells;
    ffpar.support = par.support;
    this->farField = std::make_shared<DPPoissonSlab_ns::FarField>(sys, pd, pg, ffpar);
  }
}

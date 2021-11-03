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
    
    void throwIfInvalidSplit(DPPoissonSlab::Parameters par){
      constexpr real upperRangeFactor = 3.2; //TODO: Hardcoded for 1e-3 tolerance, should change
      real minimumSplit = upperRangeFactor/(par.H + par.numberStandardDeviations*par.gw);
      real maximumSplit = 1/(par.gw*sqrt(12));
      if(par.split < minimumSplit or par.split > maximumSplit){
	System::log<System::ERROR>("Splitting parameter is outside of valid range [%g:%g] (got %g)",
				   minimumSplit, maximumSplit, par.split);
	throw std::invalid_argument("Invalid splitting parameter");
      }
    }

    real computeSplitFromNxy(DPPoissonSlab::Parameters par){
      real hxy = par.Lxy.x/par.Nxy;
      real gt = par.upsampling*hxy;
      real gw = par.gw;
      real split = sqrt(1/(4*(gt*gt - gw*gw)));
      return split;
    }

  }

  DPPoissonSlab::DPPoissonSlab(shared_ptr<ParticleGroup> pg,
			       Parameters par):
    Interactor(pg,"IBM::DPPoissonSlab"){
    if(par.split<=0){
      if(par.Nxy<=0){
	sys->log<System::EXCEPTION>("[DPPoissonSlab] I need either split or Nxy");
	  throw std::invalid_argument("Missing input parameters");
      }
      par.split = detail::computeSplitFromNxy(par);
    }
    else if(par.Nxy>0){
      sys->log<System::EXCEPTION>("[DPPoissonSlab] Pass only split OR Nxy, not both");
      throw std::invalid_argument("Invalid input parameters");
    }
    detail::throwIfInvalidSplit(par);
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
    nfpar.numberStandardDeviations = par.numberStandardDeviations;
    this->nearField = std::make_shared<DPPoissonSlab_ns::NearField>(sys, pd, pg, nfpar);
  }

  void DPPoissonSlab::initializeFarField(Parameters par){
    DPPoissonSlab_ns::FarField::Parameters ffpar;
    ffpar.split = par.split;
    ffpar.permitivity = par.permitivity;
    ffpar.gw = par.gw;
    ffpar.H = par.H;
    ffpar.numberStandardDeviations = par.numberStandardDeviations;
    ffpar.Lxy = par.Lxy;
    ffpar.tolerance = par.tolerance;
    ffpar.upsampling = par.upsampling;
    ffpar.surfaceCharge = par.surfaceCharge;
    ffpar.support = par.support;
    this->farField = std::make_shared<DPPoissonSlab_ns::FarField>(sys, pd, pg, ffpar);
  }
}

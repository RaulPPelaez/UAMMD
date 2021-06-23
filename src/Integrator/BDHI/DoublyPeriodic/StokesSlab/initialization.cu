/*Raul P. Pelaez 2019. Spectral/Chebyshev Doubly Periodic Stokes solver. Initialization functions
 */
#include"Integrator/BDHI/DoublyPeriodic/DPStokesSlab.cuh"
#include"utils.cuh"
#include"utils/cufftDebug.h"
#include"utils/cufftPrecisionAgnostic.h"
#include"misc/ChevyshevUtils.cuh"
namespace uammd{
  namespace DPStokesSlab_ns{
    DPStokes::DPStokes(DPStokes::Parameters par):
      viscosity(par.viscosity),
      H(par.H), Lx(par.Lx),Ly(par.Ly),
      tolerance(par.tolerance),
      mode(par.mode){
      setUpGrid(par);
      this->fct = std::make_shared<FastChebyshevTransform>(grid.cellDim);
      if(par.mode != WallMode::none){ //Correction is only needed for walls
	this->correction = std::make_shared<Correction>(H, make_real2(Lx, Ly), grid.cellDim, viscosity, par.mode);
      }
      initializeKernel(par);
      printStartingMessages(par);
      initializeBoundaryValueProblemSolver();
      initializeQuadratureWeights();
      precomputeIntegrals();
      CudaCheckError();
    }

    namespace DPStokes_ns{
      double proposeCellSize(real tolerance, real width){
	double h = (1.3 - std::min((-log10(tolerance))/10.0, 0.9))*width;
	return h;
      }
    }

    void DPStokes::setUpGrid(Parameters par){
      System::log<System::DEBUG>("[DPStokes] setUpGrid");
      int3 cellDim = {par.nx, par.ny, par.nz};
      if(cellDim.x < 0){
	double h = DPStokes_ns::proposeCellSize(par.tolerance, gw);
	constexpr int minimumNumberCells = 16;
	h = std::min(h, Lx/double(minimumNumberCells));
	System::log<System::MESSAGE>("[DPStokes] Proposed h: %g", h);
	cellDim = make_int3(make_real3(Lx, Ly, H)/h);
	cellDim = nextFFTWiseSize3D(cellDim);
      }
      this->grid = Grid(Box(make_real3(Lx, Ly, H)), cellDim);
      System::log<System::MESSAGE>("[DPStokes] Selected h: %g", grid.cellSize.x);
    }

    void DPStokes::initializeKernel(Parameters par){
      System::log<System::DEBUG>("[DPStokes] Initialize kernel");
      double h = grid.cellSize.x;
      // if(supportxy >= grid.cellDim.x){
      // 	System::log<System::WARNING>("[DPStokes] Support is too big, cell dims: %d %d %d, requested support: %d",
      // 				     grid.cellDim.x, grid.cellDim.y, grid.cellDim.z, supportxy);
      //}
      this->kernel = std::make_shared<Kernel>(par.w, par.beta, par.alpha, grid.cellSize.x, H, grid.cellDim.z);
      this->kernelTorque = std::make_shared<KernelTorque>(par.w_d, par.beta_d, par.alpha_d, grid.cellSize.x, H, grid.cellDim.z);
      // this->kernel = std::make_shared<Kernel>(tolerance, gw, h, H, supportxy, grid.cellDim.z);
      // this->kernelTorque = std::make_shared<KernelTorque>(tolerance, gw, h, H, supportxy, grid.cellDim.z, true);
    }

    void DPStokes::printStartingMessages(Parameters par){
      System::log<System::MESSAGE>("[DPStokes] tolerance: %g", par.tolerance);
      System::log<System::MESSAGE>("[DPStokes] support: %d", kernel->support);
      System::log<System::MESSAGE>("[DPStokes] viscosity: %g", viscosity);
      //System::log<System::MESSAGE>("[DPStokes] Gaussian source width: %g", par.gw);
      System::log<System::MESSAGE>("[DPStokes] cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
      System::log<System::MESSAGE>("[DPStokes] box size: %g %g %g", Lx, Ly, H);
      if(Lx!=Ly)
	System::log<System::WARNING>("[DPStokes] Domains with Lx=Ly are largely untested");
    }
    
    namespace detail{
      //BCs are H*du/dz(k, +-H) +- k*H*H*u(k, +-H) = RHS
      //"u" is the second integral and du/dz the first integral
      class TopBoundaryConditions{
	real k, H;
      public:
	TopBoundaryConditions(real k, real H):k(k),H(H){
	}

	real getFirstIntegralFactor() const{
	  return (k!=0)*H;
	}

	real getSecondIntegralFactor() const{
	  return k!=0?(H*H*k):(1.0);
	}
      };

      class BottomBoundaryConditions{
	real k, H;
      public:
	BottomBoundaryConditions(real k, real H):k(k),H(H){
	}

	real getFirstIntegralFactor() const{
	  return (k!=0)*H;
	}

	real getSecondIntegralFactor() const{
	  return k!=0?(-H*H*k):(1.0);
	}
      };

      template<class BoundaryConditions, class Klist>
      class BoundaryConditionsDispatch{
	Klist klist;
	real H;
      public:
	BoundaryConditionsDispatch(Klist klist, real H):klist(klist), H(H){}

	BoundaryConditions operator()(int i) const{
	  return BoundaryConditions(klist[i], H);
	}
      };

    }

    void DPStokes::initializeBoundaryValueProblemSolver(){
      System::log<System::DEBUG>("[DPStokes] Initializing BVP solver");
      const int2 nk = {grid.cellDim.x, grid.cellDim.y};
      auto klist = DPStokesSlab_ns::make_wave_vector_modulus_iterator(nk, make_real2(Lx, Ly));
      real halfH = H*0.5;
      auto topdispatch = detail::BoundaryConditionsDispatch<detail::TopBoundaryConditions, decltype(klist)>(klist, halfH);
      auto topBC = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), topdispatch);
      auto botdispatch = detail::BoundaryConditionsDispatch<detail::BottomBoundaryConditions, decltype(klist)>(klist, halfH);
      auto botBC = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), botdispatch);
      int numberSystems = (nk.x/2+1)*nk.y;
      int nz = grid.cellDim.z;
      this->bvpSolver = std::make_shared<BVP::BatchedBVPHandler>(klist, topBC, botBC, numberSystems, halfH, nz);
      CudaCheckError();
    }

    void DPStokes::initializeQuadratureWeights(){
      System::log<System::DEBUG>("[DPStokes] Initialize quadrature weights");
      real hx = grid.cellSize.x;
      real hy = grid.cellSize.y;
      int nz = grid.cellDim.z;
      qw = std::make_shared<QuadratureWeights>(H, hx, hy, nz);
    }

    namespace detail{
      namespace detail{
	enum VelOrPressure{velocity, pressure};

	std::vector<real> computeZeroModeIntegrals(real H, int nz, VelOrPressure mode){
	  std::vector<real> h_ints(nz, real());
	  constexpr int Nint = 10000;
	  std::vector<real> zwt(Nint, 0);
	  forj(0, Nint){
	    zwt[j] = 0.5*H*chebyshev::clencurt(j, Nint-1);
	  }
	  fori(0, nz){
	    forj(0, Nint){
	      real theta = M_PI*j/real(Nint-1);
	      real zcheb = cos(i*theta);
	      real Tj_z = zwt[j]*cos(theta);
	      if(mode == velocity){
		h_ints[i] += 0.5*H*Tj_z*zcheb;
	      }
	      else if( mode == pressure){
		h_ints[i] += zwt[j]*zcheb;
	      }
	    }
	  }
	  return std::move(h_ints);
	}
      }

      std::vector<real> computeZeroModePressureChebyshevIntegral(real H, int nz){
	return std::move(computeZeroModeIntegrals(H, nz, detail::VelOrPressure::pressure));
      }

      std::vector<real> computeZeroModeVelocityChebyshevIntegral(real H, int nz){
	return std::move(computeZeroModeIntegrals(H, nz, detail::VelOrPressure::velocity));
      }

    }

    void DPStokes::precomputeIntegrals(){
      System::log<System::DEBUG>("[DPStokes] Precomputing integrals");
      int nz = grid.cellDim.z;
      zeroModeVelocityChebyshevIntegrals.resize(nz);
      zeroModePressureChebyshevIntegrals.resize(nz);
      zeroModeVelocityChebyshevIntegrals = detail::computeZeroModeVelocityChebyshevIntegral(H, nz);
      zeroModePressureChebyshevIntegrals = detail::computeZeroModePressureChebyshevIntegral(H, nz);
    }

  }
}

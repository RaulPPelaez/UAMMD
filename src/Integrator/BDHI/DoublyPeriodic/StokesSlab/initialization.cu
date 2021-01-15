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
      box(par.box),
      gw(par.gw),
      tolerance(par.tolerance){
      setUpGrid(par);
      this->fct = std::make_shared<FastChebyshevTransform>(grid.cellDim);
      initializeKernel(par.support);
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
      int3 cellDim = par.cells;
      if(cellDim.x < 0){
	double h = DPStokes_ns::proposeCellSize(par.tolerance, gw);
	constexpr int minimumNumberCells = 16;
	h = std::min(h, par.box.boxSize.x/double(minimumNumberCells));
	System::log<System::MESSAGE>("[DPStokes] Proposed h: %g", h);
	cellDim = make_int3(par.box.boxSize/h);
	cellDim = nextFFTWiseSize3D(cellDim);
      }
      this->grid = Grid(par.box, cellDim);
      System::log<System::MESSAGE>("[DPStokes] Selected h: %g", grid.cellSize.x);
    }

    void DPStokes::initializeKernel(int supportxy){
      System::log<System::DEBUG>("[DPStokes] Initialize kernel");
      double h = grid.cellSize.x;
      if(supportxy >= grid.cellDim.x){
	System::log<System::WARNING>("[DPStokes] Support is too big, cell dims: %d %d %d, requested support: %d",
				     grid.cellDim.x, grid.cellDim.y, grid.cellDim.z, supportxy);
      }
      this->kernel = std::make_shared<Kernel>(tolerance, gw, h, box.boxSize.z, supportxy, grid.cellDim.z);
    }

    void DPStokes::printStartingMessages(Parameters par){
      System::log<System::MESSAGE>("[DPStokes] tolerance: %g", par.tolerance);
      System::log<System::MESSAGE>("[DPStokes] support: %d", kernel->support);
      System::log<System::MESSAGE>("[DPStokes] viscosity: %g", viscosity);
      System::log<System::MESSAGE>("[DPStokes] Gaussian source width: %g", par.gw);
      System::log<System::MESSAGE>("[DPStokes] cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
      System::log<System::MESSAGE>("[DPStokes] box size: %g %g %g", box.boxSize.x, box.boxSize.y, box.boxSize.z);
    }
    
    namespace detail{
      class TrivialBoundaryConditions{
      public:
	__host__ __device__ TrivialBoundaryConditions(int instance){}

	static real getFirstIntegralFactor(){
	  return 1.0;
	}

	static real getSecondIntegralFactor(){
	  return 1.0;
	}

      };

      template<class BoundaryConditions = TrivialBoundaryConditions>
      class BoundaryConditionsDispatch{
      public:
	__host__ __device__ BoundaryConditions operator()(int instance_index){
	  return BoundaryConditions(instance_index);
	}
      };
    }

    void DPStokes::initializeBoundaryValueProblemSolver(){
      System::log<System::DEBUG>("[DPStokes] Initializing BVP solver");
      const int2 nk = {grid.cellDim.x, grid.cellDim.y};
      const real2 Lxy = make_real2(box.boxSize);
      auto klist = DPStokesSlab_ns::make_wave_vector_modulus_iterator(nk, Lxy);
      auto topBC = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), detail::BoundaryConditionsDispatch<>());
      auto botBC = topBC;
      int numberSystems = (nk.x/2+1)*nk.y;
      real halfH = box.boxSize.z*0.5;
      int nz = grid.cellDim.z;
      this->bvpSolver = std::make_shared<BVP::BatchedBVPHandler>(klist, topBC, botBC, numberSystems, halfH, nz);
      CudaCheckError();
    }

    void DPStokes::initializeQuadratureWeights(){
      System::log<System::DEBUG>("[DPStokes] Initialize quadrature weights");
      real H = box.boxSize.z;
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
      real H = box.boxSize.z;
      zeroModeVelocityChebyshevIntegrals = detail::computeZeroModeVelocityChebyshevIntegral(H, nz);
      zeroModePressureChebyshevIntegrals = detail::computeZeroModePressureChebyshevIntegral(H, nz);
    }

  }
}

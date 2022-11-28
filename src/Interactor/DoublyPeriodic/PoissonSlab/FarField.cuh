/*Raul P. Pelaez 2020. Far field section of the Doubly Periodic Poisson solver. Slab geometry
*/
#ifndef DPPOISSONSLAB_FAR_FIELD_CUH
#define DPPOISSONSLAB_FAR_FIELD_CUH
#include "Interactor/Interactor.cuh"
#include "global/defines.h"
#include "Interactor/DoublyPeriodic/PoissonSlab/utils.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/BVPPoisson.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/spreadInterp.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/FastChebyshevTransform.cuh"
#include "Interactor/DoublyPeriodic/PoissonSlab/CorrectionCompute.cuh"
#include "utils/cufftComplex2.cuh"

namespace uammd{
  namespace DPPoissonSlab_ns{

    namespace detail{

      int3 proposeCellDim(real farFieldGaussianWidth, real upsampling, real3 L){
	constexpr int minimumNumberCells = 16; //Just an arbitrary number. It does not really help to have <=16 cells
	double h = farFieldGaussianWidth/upsampling;
	int3 cd = make_int3(L/h);
	cd.x = std::max(minimumNumberCells, cd.x);
	cd.y = std::max(minimumNumberCells, cd.y);
	//cd.z = int(M_PI/(asin(h/(L.z*0.5))) + 0.5) + 1;
	cd.z = ceil(M_PI*0.5*L.z/h);
	cd.z = std::max(minimumNumberCells, cd.z);
	//I want a number of cells in Z such that 2*Nz-2 is fft friendly
	cd.z = 2*cd.z-2;
	cd = nextFFTWiseSize3D(cd);
	while(cd.z%2 != 0){
	  cd.z++;
	  cd = nextFFTWiseSize3D(cd);
	  //cd.z must be even so that cd.z = ((2*cd.z-2)+2)/2 and 2*cd.z-2 is still a friendly number
	}
	cd.z = (cd.z+2)/2;
	return cd;
      }

      void throwIfInvalidConfiguration(double He, double minBoxSize){
	if(He > minBoxSize){
	  System::log<System::EXCEPTION>("[DPPoissonSlab] Extra height is too high (%g, max is %g), increase splitting parameter or lower tolerance", He, minBoxSize);
	  throw std::invalid_argument("[DPPoissonSlab] Incompatible parameters");
	}
      }

      double computeExtraHeight(real numberStandardDeviations, real gw, real split){
	double farGaussianWidth = sqrt(gw*gw + 1.0/(4.0*split*split));
	real He = 1.25*numberStandardDeviations*farGaussianWidth;
	return He;
      }

    }

    class FarField{
    public:
      using Grid = chebyshev::doublyperiodic::Grid;

      struct Parameters{
	real split;
        Permitivity permitivity;
	real gw;
	real H;

	real2 Lxy;
	real tolerance = -1;
	real numberStandardDeviations =-1;
	real upsampling = -1;
	int support = -1;
	bool printK0Mode = false;
	std::shared_ptr<SurfaceValueDispatch> surfaceValues = std::make_shared<SurfaceValueDispatch>();
      };

    private:
      shared_ptr<System> sys;
      shared_ptr<ParticleData> pd;
      shared_ptr<ParticleGroup> pg;
      shared_ptr<SpreadInterpolateCharges> ibm;
      shared_ptr<FastChebyshevTransform> fct;
      shared_ptr<BVPPoissonSlab> bvp;
      shared_ptr<Correction> correction;
      int3 cellDim;
      real farFieldGaussianWidth;
      real He;
      const Parameters par;
      thrust::device_vector<cufftComplex4> solutionZeroMode;
    public:

      FarField(shared_ptr<System> sys, shared_ptr<ParticleData> pd, shared_ptr<ParticleGroup> pg,
	       Parameters par):
	sys(sys), pd(pd), pg(pg), par(par){
	sys->log<System::MESSAGE>("FarField initialized");
	this->farFieldGaussianWidth = par.gw;
	if(par.split>0){
	  farFieldGaussianWidth = sqrt(par.gw*par.gw + 1.0/(4.0*par.split*par.split));
	}
	this->He = detail::computeExtraHeight(par.numberStandardDeviations, par.gw, par.split);
	const real minimumBoxSize = std::min({par.Lxy.x, par.Lxy.y, par.H});
	detail::throwIfInvalidConfiguration(He, minimumBoxSize);
	real3 totalBoxSize = make_real3(par.Lxy, par.H + 6*He);
	this->cellDim = detail::proposeCellDim(farFieldGaussianWidth, par.upsampling, totalBoxSize);
	sys->log<System::MESSAGE>("[DPPoissonSlab] cells: %d %d %d", cellDim.x, cellDim.y, cellDim.z);
	sys->log<System::MESSAGE>("[DPPoissonSlab] box size: %g %g %g (enlarged to %g)",
				  par.Lxy.x, par.Lxy.y, par.H, par.H + He*6);
	sys->log<System::MESSAGE>("[DPPoissonSlab] Extra height: %g", He);
	if(par.split> 0){
	  sys->log<System::MESSAGE>("[DPPoissonSlab] Ewald split mode enabled");
	  sys->log<System::MESSAGE>("[DPPoissonSlab] split: %g", par.split);
	  sys->log<System::MESSAGE>("[DPPoissonSlab] Far field width: %g, (%g times original width)",
				    farFieldGaussianWidth, farFieldGaussianWidth/par.gw);
	}
	this->ibm = createImmersedBoundary();
	this->fct = std::make_shared<FastChebyshevTransform>(sys, cellDim);
	this->bvp = createBVP(par);
	this->correction = std::make_shared<Correction>(par.permitivity, cellDim, par.Lxy, par.H, He);
	setUpSurfaceValues(par.surfaceValues);
      }

      void compute(cudaStream_t st, real4 *fieldAtParticles = nullptr){
	System::log<System::DEBUG>("Far field");
	auto separatedCharges = separateCharges(st); //Step 1
	auto gridCharges = ibm->spreadChargesNearWalls(separatedCharges, st); //Step 2
	//DPPoissonSlab_ns::writeRealField<real>(gridCharges, cellDim, "gf_C1.dat");
	auto gridChargesFourier = fct->forwardTransform(gridCharges, st);//Step 2
	auto outsideSolution = bvp->solveFieldPotential(gridChargesFourier, st); //Step 2
	//DPPoissonSlab_ns::writeComplex4Field(outsideSolution, cellDim, par.H,"solOut.dat");
	ibm->spreadChargesFarFromWallAdd(separatedCharges, gridCharges, st); //Step 3
	ibm->spreadImageChargesAdd(separatedCharges, gridCharges, par.permitivity, st); //Step 3
	//DPPoissonSlab_ns::writeRealField<real>(gridCharges, cellDim, "gf_C1pC2pC1star.dat");
	gridChargesFourier = fct->forwardTransform(gridCharges, st); //Step 3
	//DPPoissonSlab_ns::writeComplexField(gridChargesFourier, cellDim, par.H, "gf_C1pC2pC1starFourier.dat");
	auto insideSolution = bvp->solveFieldPotential(gridChargesFourier, st); //Step 3
	//DPPoissonSlab_ns::writeComplex4Field(insideSolution, cellDim, par.H,"solIn.dat");
	auto surfaceValues_ptr = thrust::raw_pointer_cast(surfaceValuesFourier.data());
	correction->correctSolution(insideSolution, outsideSolution, surfaceValues_ptr, st); //Steps 4,5,6
	//DPPoissonSlab_ns::writeComplex4Field(insideSolution, cellDim, par.H,"solCorrected.dat");
	if(par.printK0Mode){
	  solutionZeroMode.resize(cellDim.z);
	  Index3D indexer(cellDim.x/2+1, cellDim.y, cellDim.z);
	  auto k0z = make_third_index_iterator(insideSolution.begin(), 0,0, indexer);
	  thrust::copy(thrust::cuda::par.on(st),
		       k0z, k0z + cellDim.z,
		       solutionZeroMode.begin());
	}
	auto gridFields = fct->inverseTransform(insideSolution, st);
	//DPPoissonSlab_ns::writeRealField<real4>(gridFields, cellDim, "solGrid.dat");
	ibm->interpolateFieldsToParticles(gridFields, st, fieldAtParticles);
	CudaCheckError();
      }

      void setSurfaceValuesZeroModeFourier(cufftComplex2 zeroMode){
	surfaceValuesFourier[0] = zeroMode*cellDim.x*cellDim.y;
      }

      auto getK0Mode(){
	System::log<System::DEBUG1>("[DPPoissonSlab] Downloading zero mode data");
	std::vector<cufftComplex4> h_zeroMode(solutionZeroMode.size());
	thrust::copy(solutionZeroMode.begin(), solutionZeroMode.end(), h_zeroMode.begin());
	return h_zeroMode;
      }
    private:

      std::shared_ptr<SpreadInterpolateCharges> createImmersedBoundary(){
	SpreadInterpolateCharges::Parameters ibmpar;
	real3 L = make_real3(par.Lxy, par.H + 6*He);
	ibmpar.grid = Grid(Box(L), cellDim);
	ibmpar.H = par.H;
	ibmpar.He = He;
	ibmpar.gaussianWidth = farFieldGaussianWidth;
	ibmpar.tolerance = par.tolerance;
	ibmpar.support = par.support;
	if(par.split>0){
	  real Htot = par.H + 6*He;
	  int czmax = int((cellDim.z-1)*(acos(2.0*(0.5*par.H + 2*He)/Htot)/real(M_PI)));
	  ibmpar.maximumSupport = 2*czmax+1;
	}
	else{
	  ibmpar.maximumSupport = cellDim.z;
	}
	return std::make_shared<SpreadInterpolateCharges>(sys, pd, ibmpar);
      }

      SeparatedCharges separateCharges(cudaStream_t st){
	SeparatedCharges sep;
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	sep.separate(pos, par.H, 2*He, st);
	return sep;
      }

      std::shared_ptr<BVPPoissonSlab> createBVP(Parameters par){
	BVPPoissonSlab::Parameters bvppar;
	bvppar.Lxy = par.Lxy;
	bvppar.H = 0.5*(par.H + 6*He);
	bvppar.cellDim = cellDim;
	bvppar.permitivity = par.permitivity.inside;
	return std::make_shared<BVPPoissonSlab>(bvppar);
      }

      void setUpSurfaceValues(std::shared_ptr<SurfaceValueDispatch> surfaceValue);

      gpu_container<cufftComplex2> surfaceValuesFourier;
    };

    gpu_container<cufftComplex2> takeSurfaceValueDensityToFourier(cached_vector<real> &surfaceValueTop,
								  cached_vector<real> &surfaceValueBottom, int2 gridSize){
      int2 n = gridSize;
      gpu_container<cufftComplex2> surfaceValuesFourier(n.y*(n.x/2+1));
      int2 cdtmp = {n.y, n.x};
      int2 inembed = {n.y, 2*(n.x/2+1)};
      int2 oembed = {n.y, n.x/2+1};
      cufftHandle cufft_plan_forward;
      CufftSafeCall(cufftPlanMany(&cufft_plan_forward,
				  2, &cdtmp.x,
				  &inembed.x,
				  1, 1,
				  &oembed.x,
				  2, 1,
				  CUFFT_Real2Complex<real>::value, 1));
      real* i_data = thrust::raw_pointer_cast(surfaceValueTop.data());
      cufftComplex* o_data = (cufftComplex*)thrust::raw_pointer_cast(surfaceValuesFourier.data());
      CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, i_data, o_data));
      i_data = thrust::raw_pointer_cast(surfaceValueBottom.data());
      CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, i_data, o_data+1));
      return surfaceValuesFourier;
    }

    void FarField::setUpSurfaceValues(std::shared_ptr<SurfaceValueDispatch> surfaceValue){
      int2 n = {cellDim.x,cellDim.y};
      auto charges = pd->getCharge(access::location::cpu, access::mode::read);
      real qtot = std::accumulate(charges.begin(), charges.end(), real());
      std::vector<real> surfaceValueTop(2*(n.x/2+1)*n.y);
      auto surfaceValueBottom = surfaceValueTop;
      forj(0, n.y){
	fori(0, n.x){
	  real x = (i/real(n.x)-0.5)*par.Lxy.x;
	  real y = (j/real(n.y)-0.5)*par.Lxy.y;
	  surfaceValueTop[i + 2*(n.x/2+1)*j] = surfaceValue->top(x,y);
	  surfaceValueBottom[i + 2*(n.x/2+1)*j] = surfaceValue->bottom(x,y);
	}
      }
      real totalWallCharge = std::accumulate(surfaceValueTop.begin(), surfaceValueTop.end(), 0.0);
      totalWallCharge += std::accumulate(surfaceValueBottom.begin(), surfaceValueBottom.end(), 0.0);
      real totalSystemCharge =  qtot + totalWallCharge;
      if(totalSystemCharge != 0){
	System::log<System::WARNING>("[DPPoissonSlab] The system is not electroneutral (found %g total charge)", totalSystemCharge);
	System::log<System::WARNING>("[DPPoissonSlab] To ensure electroneutrality half an opposite charge will be placed on each wall. This is an assumption of the algorithm and cannot be avoided.");
      }
      cached_vector<real> sct(surfaceValueTop);
      cached_vector<real> scb(surfaceValueBottom);
      surfaceValuesFourier = takeSurfaceValueDensityToFourier(sct, scb, n);
    }

  }
}
#endif

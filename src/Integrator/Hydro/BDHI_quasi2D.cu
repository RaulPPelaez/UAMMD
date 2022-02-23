/*Raul P. Pelaez 2019-2020. Quasi2D Integrator
  See BDHI_quasi2D.cuh for more information.
 */
#include"BDHI_quasi2D.cuh"
#include"utils/GPUUtils.cuh"
#include"third_party/saruprng.cuh"
#include<vector>
#include<algorithm>
#include<fstream>
#include"utils/debugTools.h"
#include"utils/cufftDebug.h"
#include"utils/cxx_utils.h"
#include"third_party/type_names.h"
#include<fstream>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/tuple.h>
namespace uammd{
  namespace BDHI{

    template<class HydroKernel>
    BDHI2D<HydroKernel>::BDHI2D(shared_ptr<ParticleGroup> pg,
				Parameters par):
      Integrator(pg,"BDHI::BDHI2D"),
      dt(par.dt),
      temperature(par.temperature),
      tolerance(par.tolerance),
      viscosity(par.viscosity),
      box(par.box), grid(par.box, int3()), hydroKernel(par.hydroKernel),
      hydrodynamicRadius(par.hydrodynamicRadius){
      if(!hydroKernel){
	hydroKernel = std::make_shared<HydroKernel>();
      }
      seed = sys->rng().next32();
      checkInputParametersValidity(par);
      box = Box(make_real3(par.box.boxSize.x, par.box.boxSize.y, 0));
      initializeGrid(par);
      initializeInterpolationKernel(par);
      printStartingMessages();
      resizeContainers();
      initCuFFT();
      CudaSafeCall(cudaStreamCreate(&st));
      CudaSafeCall(cudaStreamCreate(&st2));
      // st = 0;
      // st2 = 0;
      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::checkInputParametersValidity(Parameters par){
      if(par.box.boxSize.x == real(0.0) && par.box.boxSize.y == real(0.0)){
	sys->log<System::ERROR>("[BDHI::BDHI2D] Box of size zero detected, cannot work without a box! (make sure a box parameter was passed)");
	throw std::runtime_error("Invalid box");
      }
      if(hydrodynamicRadius<=0){
	sys->log<System::ERROR>("[BDHI::BDHI2D] Please specify a valid hydrodynamic radius");
	throw std::runtime_error("Invalid hydrodynamic radius");
      }
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::initializeGrid(Parameters par){
      int3 cellDim;
      if(par.cells.x <= 0){
	const double upsampling = 0.8;
	const double h = hydrodynamicRadius*upsampling;
        cellDim = nextFFTWiseSize3D(make_int3(box.boxSize/h));
	cellDim.z = 1;
      }
      else{
        cellDim = make_int3(par.cells, 1);
      }
      grid = Grid(box, cellDim);
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::initializeInterpolationKernel(Parameters par){
      int support  = (int(3.0*hydrodynamicRadius*grid.cellDim.x/grid.box.boxSize.x) + 1 )*2+1;
      if(support > grid.cellDim.x){
	support = grid.cellDim.x;
      }
      double width = hydroKernel->getGaussianVariance(par.hydrodynamicRadius);
      this->ibmKernel = std::make_shared<Kernel>(support, width);
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::resizeContainers(){
      try{
	const int ncells = (grid.cellDim.x/2+1)*grid.cellDim.y;
	gridVelsFourier.resize(ncells, cufftComplex2());
	gridVels.resize(2*ncells, real2());
	const int numberParticles = pg->getNumberParticles();
	particleVels.resize(numberParticles, real2());
      }
      catch(...){
	sys->log<System::ERROR>("[BDHI::BDHI2D] Could not resize containers");
	throw;
      }
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::initCuFFT(){
      CufftSafeCall(cufftCreate(&cufft_plan_forward));
      CufftSafeCall(cufftCreate(&cufft_plan_inverse));
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));
      size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
      //This sizes have to be reversed according to the cufft docs
      const auto n = grid.cellDim;
      int2 cdtmp = {n.y, n.x};
      int2 inembed = {n.y, 2*(n.x/2+1)};
      int2 oembed = {n.y, n.x/2+1};
      CufftSafeCall(cufftMakePlanMany(cufft_plan_forward,
				      2, &cdtmp.x,
				      &inembed.x,
				      2, 1,
				      &oembed.x,
				      2, 1,
				      CUFFT_Real2Complex<real>::value, 2,
				      &cufftWorkSizef));
      CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
				      2, &cdtmp.x,
				      &oembed.x,
				      2, 1,
				      &inembed.x,
				      2, 1,
				      CUFFT_Complex2Real<real>::value, 2,
				      &cufftWorkSizei));
      size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei)+10;
      size_t free_mem, total_mem;
      CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
      sys->log<System::DEBUG>("[BDHI::BDHI2D] cuFFT grid size: %d %d", cdtmp.x, cdtmp.y);
      sys->log<System::DEBUG>("[BDHI::BDHI2D] Necessary work space for cuFFT: %s, available: %s, total: %s",
			      printUtils::prettySize(cufftWorkSize).c_str(),
			      printUtils::prettySize(free_mem).c_str(),
			      printUtils::prettySize(total_mem).c_str());
      try{
	cufftWorkArea.resize(cufftWorkSize);
      }
      catch(...){
	sys->log<System::ERROR>("[BDHI::BDHI2D] Not enough memory in device to allocate cuFFT free %s, needed: %s!!",
				printUtils::prettySize(free_mem).c_str(),
				printUtils::prettySize(cufftWorkSize).c_str());
	throw;
      }
      auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::printStartingMessages(){
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Initialized");
      sys->log<System::WARNING>("[BDHI::BDHI2D] The tolerance parameter is currently ignored. Tolerance is hardcoded to ~1e-4");
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Temperature: %g", temperature);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Viscosity: %g", viscosity);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] dt: %g", dt);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] hydrodynamic radius: %g", hydrodynamicRadius);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Using interpolation kernel: %s",
				type_name_without_namespace<Kernel>().c_str());
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Using hydrodynamic kernel: %s",
				type_name_without_namespace<HydroKernel>().c_str());
      if(box.boxSize.x != box.boxSize.y){
	sys->log<System::WARNING>("[BDHI::BDHI2D] Self mobility will be different for non square boxes!");
      }
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Box Size: %g %g", grid.box.boxSize.x, grid.box.boxSize.y);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Grid dimensions: %d %d", grid.cellDim.x, grid.cellDim.y);
      double rh = hydrodynamicRadius;
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Interpolation kernel support: %g rh max distance, %d cells total",
				ibmKernel->support*0.5*grid.cellSize.x/rh, ibmKernel->support);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Gaussian variance: %g", hydroKernel->getGaussianVariance(rh));
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] h: %g %g", grid.cellSize.x, grid.cellSize.y);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Cell volume: %e", grid.cellSize.x*grid.cellSize.y);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Requested kernel tolerance: %g", tolerance);
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::forwardTime(){
      for(auto updatable: updatables){
	updatable->updateSimulationTime(step*dt);
      }
      step++;
      if(step==1){
	for(auto updatable: updatables){
	  updatable->updateTemperature(temperature);
	  updatable->updateBox(box);
	  updatable->updateTimeStep(dt);
	  updatable->updateViscosity(viscosity);
	}
      }
      sys->log<System::DEBUG1>("[BDHI::BDHI2D] Performing step");
      resetContainers();
      for(auto inter: interactors){
	inter->sum({.force = true, .energy= false, .virial = false}, st2);
      }
      CudaSafeCall(cudaStreamSynchronize(st2));
      spreadParticles();
      convolveFourier();
      interpolateVelocities();
      updateParticlePositions();
      CudaSafeCall(cudaStreamSynchronize(st));
      CudaCheckError();
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Done");
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::resetContainers(){
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Setting vels to zero...");
      thrust::fill(thrust::cuda::par.on(st2), particleVels.begin(), particleVels.end(), real2());
      if(interactors.size() > 0 or (hydroKernel->hasThermalDrift() and temperature > 0)){
	thrust::fill(thrust::cuda::par.on(st), gridVels.begin(), gridVels.end(), typename decltype(gridVels)::value_type());
      }
      if(interactors.size() > 0 or temperature > 0){
	thrust::fill(thrust::cuda::par.on(st), gridVelsFourier.begin(), gridVelsFourier.end(), typename decltype(gridVelsFourier)::value_type());
      }
      auto force = pd->getForce(access::location::gpu, access::mode::write);
      thrust::fill(thrust::cuda::par.on(st2), force.begin(), force.end(), real4());
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::spreadParticles(){
      spreadThermalDrift();
      spreadParticleForces();
      CudaSafeCall(cudaStreamSynchronize(st));
      CudaCheckError();
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::spreadThermalDrift(){
      if(hydroKernel->hasThermalDrift() and temperature > 0){
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Spreading thermal drift");
	const int numberParticles = pg->getNumberParticles();
	const auto pos = pd->getPos(access::location::gpu, access::mode::read);
	real2* d_gridVels = thrust::raw_pointer_cast(gridVels.data());
	const auto n = grid.cellDim;
	double width = hydroKernel->getGaussianVariance(hydrodynamicRadius);
	const auto trX = thrust::make_constant_iterator<real2>({-temperature,0});
	auto kernelX = std::make_shared<KernelThermalDrift<0>>(ibmKernel->support, width);
	IBM<KernelThermalDrift<0>> ibmX(kernelX, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
	ibmX.spread(pos.begin(), trX, d_gridVels, numberParticles, st);
	const auto trY = thrust::make_constant_iterator<real2>({0,-temperature});
	auto kernelY = std::make_shared<KernelThermalDrift<1>>(ibmKernel->support, width);
	IBM<KernelThermalDrift<1>> ibmY(kernelY, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
	ibmY.spread(pos.begin(), trY, d_gridVels, numberParticles, st);
	CudaCheckError();
      }
    }

    namespace BDHI2D_ns{
      struct toReal2{
	template<class vtype> inline __device__ real2 operator()(vtype q){ return make_real2(q);}
      };
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::spreadParticleForces(){
      if(interactors.size()>0){
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Spread particle forces");
	const int numberParticles = pg->getNumberParticles();
	const auto pos = pd->getPos(access::location::gpu, access::mode::read);
	real2* d_gridVels = (real2*)thrust::raw_pointer_cast(gridVels.data());
	const auto force = pd->getForce(access::location::gpu, access::mode::read);
	const auto f_tr = thrust::make_transform_iterator(force.begin(), BDHI2D_ns::toReal2());
	const auto n = grid.cellDim;
	IBM<Kernel> ibm(ibmKernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
	ibm.spread(pos.begin(), f_tr, d_gridVels, numberParticles, st);
	CudaCheckError();
      }
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::convolveFourier(){
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Wave space convolution");
      //In: S·F -> Out: FFT·S·F
      forwardTransformVelocities();
      //In: FFT·S·F -> Out: B·FFT·S·F
      applyGreenFunctionConvolutionFourier();
      //In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw
      addStochastichTermFourier();
      //In: B·FFT·S·F + 1/√σ·√B·dWw -> Out FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )
      inverseTransformVelocities();
      CudaCheckError();
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::forwardTransformVelocities(){
      bool isGridDataNonZero = interactors.size() > 0 or (hydroKernel->hasThermalDrift() and temperature > 0);
      if(isGridDataNonZero){
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Taking grid to wave space");
	CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
	auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
	auto d_gridVelsF = thrust::raw_pointer_cast(gridVelsFourier.data());
	CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, (cufftReal*)d_gridVels, (cufftComplex*)d_gridVelsF));
      }
    }

    namespace BDHI2D_ns{

      using cufftComplex2 = BDHI2D_ns::cufftComplex2;
      using cufftComplex = BDHI2D_ns::cufftComplex;
      using cufftReal = BDHI2D_ns::cufftReal;

      template<class vec2>
      inline __device__ vec2 cellToWaveNumber(int2 cell, int2 cellDim, vec2 L){
	const vec2 pi2invL = (real(2.0)*real(M_PI))/L;
	vec2 k = {(cell.x - cellDim.x*(cell.x >= (cellDim.x/2+1)))*pi2invL.x,
	 	  (cell.y - cellDim.y*(cell.y >= (cellDim.y/2+1)))*pi2invL.y};
	return k;
      }

      //Performs the product G_k*factor_k, given f_k and g_k, see Algorithm 1.3 in [1]
      inline __device__ cufftComplex2 projectFourier(real2 k, cufftComplex2 factor,
						     real fk, real gk){
	cufftComplex2 res;
	{//Real part
	  const real2 fr = make_real2(factor.x.x, factor.y.x);
	  const real2 vr = { k.y*gk*dot(fr, {k.y, -k.x}) + k.x*fk*dot(fr, k),
			     -k.x*gk*dot(fr, {k.y, -k.x}) + k.y*fk*dot(fr, k)};
	  res.x.x = vr.x;
	  res.y.x = vr.y;
	}
	{//Imaginary part
	  const real2 fi = make_real2(factor.x.y, factor.y.y);
	  const real2 vi = { k.y*gk*dot(fi, {k.y, -k.x}) + k.x*fk*dot(fi, k),
			     -k.x*gk*dot(fi, {k.y, -k.x}) + k.y*fk*dot(fi, k)};
	  res.x.y = vi.x;
	  res.y.y = vi.y;
	}
	return res;
      }

      template<class HydroKernel>
      __global__ void forceFourier2Vel(cufftComplex2 * gridForces,
				       cufftComplex2 * gridVels,
				       real vis,
				       Grid grid,
				       HydroKernel hydroKernel,
				       real hydrodynamicRadius){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	const int2 nk = make_int2(grid.cellDim);
	if(id >= nk.y*(nk.x/2+1)){
	  return;
	}
	int2 ik = make_int2(id%(nk.x/2+1), id/(nk.x/2+1));
	if(id == 0){
	  gridVels[0] = cufftComplex2();
	  return;
	}
	const int ncells = grid.getNumberCells();
	real2 k = cellToWaveNumber(ik, nk, make_real2(grid.box.boxSize));
	const real k2 = dot(k, k);
	const real a = hydrodynamicRadius;
	const real2 fg = hydroKernel(k2, a);
	const real fk = fg.x/(vis*real(nk.x*nk.y));
	const real gk = fg.y/(vis*real(nk.x*nk.y));
	const cufftComplex2 factor = gridForces[id];
	gridVels[id] = projectFourier(k, factor, fk, gk);
      }

      template<class HydroKernel>
      __global__ void fourierBrownianNoise(cufftComplex2 *__restrict__ gridVelsFourier,
					   Grid grid,
					   real prefactor,
					   real vis,
					   ullint seed,
					   ullint step,
					   HydroKernel hydroKernel,
					   real hydrodynamicRadius){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	const int2 nk = make_int2(grid.cellDim);
	if(id >= nk.y*(nk.x/2+1)){
	  return;
	}
	int2 ik = make_int2(id%(nk.x/2+1), id/(nk.x/2+1));
	if(id == 0){
	  gridVelsFourier[0] = cufftComplex2();
	  return;
	}
	if(ik.x == 0 and ik.y > (nk.y - ik.y)) return;
	if(ik.x == nk.x - ik.x and ik.y > (nk.y - ik.y) ) return;
	const bool isXnyquist = (ik.x == (nk.x - ik.x)) and (nk.x%2 == 0);
	const bool isYnyquist = (ik.y == (nk.y - ik.y)) and (nk.y%2 == 0);
	if(isXnyquist and ik.y == 0){
	  gridVelsFourier[id] = cufftComplex2();
	}
	const bool isNyquist = (isYnyquist and ik.x == 0) or (isXnyquist and isYnyquist );
	cufftComplex2 noise;
	{
	  Saru saru(id, step, seed);
	  const real complex_gaussian_sc = real(0.707106781186547)*prefactor; //1/sqrt(2)
	  noise.x = make_real2(saru.gf(0, complex_gaussian_sc));
	  noise.y = make_real2(saru.gf(0, complex_gaussian_sc));
	  if(isNyquist){
	    noise.x.x *= real(1.41421356237310);
	    noise.y.x *= real(1.41421356237310);
	    noise.x.y = 0;
	    noise.y.y = 0;
	  }
	}
	{
	  const real2 k = cellToWaveNumber(ik, nk, make_real2(grid.box.boxSize));
	  cufftComplex2 factor;
	  {
	    const real k2 = dot(k,k);
	    const real a = hydrodynamicRadius;
	    const real2 fg = hydroKernel(k2, a);
	    const real fk = fg.x;
	    const real gk = fg.y;
	    const real fk_sq = sqrt(fk);
	    const real gk_sq = sqrt(gk);
	    factor.x = (gk_sq*noise.x*k.y    + fk_sq*noise.y*k.x);
	    factor.y = (gk_sq*noise.x*(-k.x) + fk_sq*noise.y*k.y);
	  }
	  gridVelsFourier[id] += factor;
	  if(isNyquist) return;
	  //Ensure correct conjugacy v_i = \conj{v_{N-i}}
	  if(ik.x == (nk.x-ik.x) or ik.x == 0){
	    factor.x.y *= real(-1.0);
	    factor.y.y *= real(-1.0);
	    const int indexOfConjugate =  ik.x + (nk.x/2 + 1)*(nk.y - ik.y);
	    gridVelsFourier[indexOfConjugate] += factor;
	  }
	}
      }

    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::applyGreenFunctionConvolutionFourier(){
      bool isGridDataNonZero = interactors.size() > 0 or (hydroKernel->hasThermalDrift() and temperature > 0);
      if(isGridDataNonZero){
	auto d_gridVels = (cufftComplex2*)thrust::raw_pointer_cast(gridVelsFourier.data());
	const int nthreads = 128;
	const int nblocks = ((grid.cellDim.x/2+1)*grid.cellDim.y)/128+1;
	BDHI2D_ns::forceFourier2Vel<<<nblocks, nthreads, 0, st>>>(d_gridVels,
								  d_gridVels,
								  viscosity,
								  grid,
								  *hydroKernel,
								  hydrodynamicRadius);
      }
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::addStochastichTermFourier(){
      /*Add if T>0 -> 1/√σ·√B·dWw */
      if(temperature > 0){
	auto d_gridVels = (cufftComplex2*)thrust::raw_pointer_cast(gridVelsFourier.data());
	static ullint counter = 0;
	counter++;
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Wave space brownian noise");
	const real prefactor = sqrt(2.0*temperature/(viscosity*dt*box.boxSize.x*box.boxSize.y));
	const int nthreads = 128;
	const int nblocks = ((grid.cellDim.x/2+1)*grid.cellDim.y)/nthreads+1;
	BDHI2D_ns::fourierBrownianNoise<<<nblocks, nthreads, 0, st>>>(d_gridVels,
								      grid,
								      prefactor,
								      viscosity,
								      seed,
								      counter,
								      *hydroKernel,
								      hydrodynamicRadius);
      }
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::inverseTransformVelocities(){
      bool isGridDataNonZero = interactors.size() > 0 or temperature > 0;
      if(isGridDataNonZero){
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Going back to real space");
	CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));
	auto d_gridVelsF = thrust::raw_pointer_cast(gridVelsFourier.data());
	auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
	CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, (cufftComplex*)d_gridVelsF, (cufftReal*)d_gridVels));
      }
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::interpolateVelocities(){
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Interpolate grid velocities to particles");
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      real2* d_gridVels = (real2*)thrust::raw_pointer_cast(gridVels.data());
      real2* d_particleVels = thrust::raw_pointer_cast(particleVels.data());
      const auto n = grid.cellDim;
      IBM<Kernel> ibm(ibmKernel, grid, IBM_ns::LinearIndex3D(2*(n.x/2+1), n.y, n.z));
      ibm.gather(pos.begin(), d_particleVels, d_gridVels, numberParticles, st);
      CudaCheckError();
    }

    namespace BDHI2D_ns{
      class euler_functor: public thrust::binary_function<real2, real4, real4>{
	real dt;
      public:
	euler_functor(real _dt) : dt(_dt) {}

        __host__ __device__ real4 operator()(real2 vel, real4 pos) const {
	  return pos + make_real4(vel*dt);
        }

      };
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::updateParticlePositions(){
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      const int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Updating positions");
      try{
	thrust::transform(thrust::cuda::par.on(st),
			  particleVels.begin(), particleVels.end(),
			  pos.begin(),
			  pos.begin(),
			  BDHI2D_ns::euler_functor(dt));
      }
      catch(thrust::system_error &e){
	sys->log<System::ERROR>("[BDHI::BDHI2D] Thrust transform failed with: %s", e.what());
	throw;
      }
      CudaCheckError();
    }

    template<class HydroKernel>
    BDHI2D<HydroKernel>::~BDHI2D(){
      try{
	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	CufftSafeCall(cufftDestroy(cufft_plan_inverse));
	CufftSafeCall(cufftDestroy(cufft_plan_forward));
	CudaSafeCall(cudaStreamDestroy(st));
	CudaSafeCall(cudaStreamDestroy(st2));
      }
      catch(...){
	sys->log<System::ERROR>("[BDHI::BDHI2D] Exception raised in BDHI2D destructor, silencing...");
      }
    }

  }
}

/*Raul P. Pelaez 2019. Quasi2D Integrator
  See BDHI_quasi2D.cuh for more information.
 */
#include"BDHI_quasi2D.cuh"
#include"utils/GPUUtils.cuh"
#include"third_party/saruprng.cuh"
#include<vector>
#include<algorithm>
#include<fstream>
#include"utils/debugTools.cuh"
#include"utils/cufftDebug.h"
#include"utils/cxx_utils.h"
#include"third_party/type_names.h"
#include<fstream>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/tuple.h>
namespace uammd{
  namespace BDHI{
    template<class HydroKernel>
    BDHI2D<HydroKernel>::BDHI2D(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys,
	     Parameters par):
      Integrator(pd, pg, sys, "BDHI::BDHI2D"),
      dt(par.dt),
      temperature(par.temperature),
      viscosity(par.viscosity),
      box(par.box), grid(par.box, int3()), hydroKernel(par.hydroKernel),
      hydrodynamicRadius(par.hydrodynamicRadius){
      sys->log<System::WARNING>("[BDHI::BDHI2D] The tolerance parameter is currently ignored. Tolerance is hardcoded to ~1e-4");
      if(!hydroKernel) hydroKernel = std::make_shared<HydroKernel>();
      seed = sys->rng().next();
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Initialized");
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Temperature: %g", temperature);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Viscosity: %g", viscosity);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] dt: %g", dt);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] hydrodynamic radius: %g", hydrodynamicRadius);
      
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Using interpolation kernel: %s",
				type_name_without_namespace<Kernel>().c_str());
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Using hydrodynamic kernel: %s",
				type_name_without_namespace<HydroKernel>().c_str());
      if(box.boxSize.x == real(0.0) && box.boxSize.y == real(0.0)){
	sys->log<System::CRITICAL>("[BDHI::BDHI2D] Box of size zero detected, cannot work without a box! (make sure a box parameter was passed)");
      }
      if(hydrodynamicRadius<=0) sys->log<System::CRITICAL>("[BDHI::BDHI2D] Please specify a valid hydrodynamic radius");
      box = Box(make_real3(box.boxSize.x, box.boxSize.y, 0));

      int numberParticles = pg->getNumberParticles();

      int3 cellDim;
      if(par.cells.x<=0){
	real h = par.hydrodynamicRadius*0.8;
	cellDim = nextFFTWiseSize3D(make_int3(box.boxSize/h));
      }
      else{
        cellDim = make_int3(par.cells,1);
      }
      cellDim.z = 1;
      grid = Grid(box, cellDim);

      int support  = (int(2.5 * par.hydrodynamicRadius * grid.cellDim.x / grid.box.boxSize.x) +1 )*2+1;

      double width = hydroKernel->getGaussianVariance(par.hydrodynamicRadius);
      auto kernel = std::make_shared<Kernel>(support, width);
      ibm = std::make_shared<IBM<Kernel>>(sys, kernel, grid);
      {
	auto kernelThermalDrift = std::make_shared<KernelThermalDrift>(support, width);
	ibmThermalDrift = std::make_shared<IBM<KernelThermalDrift>>(sys, kernelThermalDrift, grid);
      }


      //Try to set the closest rh possible
      if(par.cells.x<=0){
       	double h = grid.cellSize.x;
       	cellDim = nextFFTWiseSize3D(make_int3(box.boxSize/h));
	cellDim.z = 1;
       	grid = Grid(box, cellDim);
	kernel = std::make_shared<Kernel>(support, width);
	ibm = std::make_shared<IBM<Kernel>>(sys, kernel, grid);
      }
      double rh = par.hydrodynamicRadius;

      if(box.boxSize.x != box.boxSize.y){
	sys->log<System::WARNING>("[BDHI::BDHI2D] Self mobility will be different for non square boxes!");
      }

      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Box Size: %g %g", grid.box.boxSize.x, grid.box.boxSize.y);

      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Grid dimensions: %d %d", grid.cellDim.x, grid.cellDim.y);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Interpolation kernel support: %g rh max distance, %d cells total", kernel->support*0.5*grid.cellSize.x/rh, kernel->support);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Gaussian variance: %g", width);

      sys->log<System::MESSAGE>("[BDHI::BDHI2D] h: %g %g", grid.cellSize.x, grid.cellSize.y);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Cell volume: %e", grid.cellSize.x*grid.cellSize.y);
      sys->log<System::MESSAGE>("[BDHI::BDHI2D] Requested kernel tolerance: %g", par.tolerance);

      if(kernel->support >= grid.cellDim.x or
	 kernel->support >= grid.cellDim.y)
	sys->log<System::CRITICAL>("[BDHI::BDHI2D] Kernel support is too big, try lowering the tolerance or increasing the box size!.");

      CudaSafeCall(cudaStreamCreate(&st));
      CudaSafeCall(cudaStreamCreate(&st2));


      /*The quantity spreaded to the grid in real or wave space*/
      /*The layout of this array is
	fx000, fy000, fz000, fx001, fy001, fz001..., fxnnn, fynnn, fznnn. n=ncells-1
	When used in real space each f is a real number, whereas in wave space each f will be a complex number.
	See cufftC2R of R2C in place in Mdot_far
      */
      /*Can be Force when spreading particles to the grid and
	velocities when interpolating from the grid to the particles*/
      int ncells = grid.cellDim.x*grid.cellDim.y;
      gridVelsFourier.resize(2*ncells, cufftComplex());
      particleVels.resize(numberParticles, real2());
      initCuFFT();

      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::initCuFFT(){
      CufftSafeCall(cufftCreate(&cufft_plan_forward));
      CufftSafeCall(cufftCreate(&cufft_plan_inverse));

      /*I will be handling workspace memory*/
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));

      //Required storage for the plans
      size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
      /*Set up cuFFT*/
      //This sizes have to be reversed according to the cufft docs
      int2 cdtmp = {grid.cellDim.y, grid.cellDim.x};
      int2 inembed = {grid.cellDim.y, grid.cellDim.x};
      /*I want to make three 3D FFTs, each one using one of the three interleaved coordinates*/
      CufftSafeCall(cufftMakePlanMany(cufft_plan_forward,
				      2, &cdtmp.x,
				      &inembed.x,
				      2, 1,
				      &inembed.x,
				      2, 1,
				      CUFFT_Real2Complex<real>::value, 2,
				      &cufftWorkSizef));

      sys->log<System::DEBUG>("[BDHI::BDHI2D] cuFFT grid size: %d %d", cdtmp.x, cdtmp.y);
      /*Same as above, but with C2R for inverse FFT*/
      CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
				      2, &cdtmp.x,
				      &inembed.x,
				      2, 1,
				      &inembed.x,
				      2, 1,
				      CUFFT_Complex2Real<real>::value, 2,
				      &cufftWorkSizei));

      /*Allocate cuFFT work area*/
      size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei)+10;
      size_t free_mem, total_mem;
      CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));

      sys->log<System::DEBUG>("[BDHI::BDHI2D] Necessary work space for cuFFT: %s, available: %s, total: %s",
			      printUtils::prettySize(cufftWorkSize).c_str(),
			      printUtils::prettySize(free_mem).c_str(),
			      printUtils::prettySize(total_mem).c_str());

      if(free_mem<cufftWorkSize){
	sys->log<System::CRITICAL>("[BDHI::BDHI2D] Not enough memory in device to allocate cuFFT free %s, needed: %s!!",
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(cufftWorkSize).c_str());
      }

      cufftWorkArea.resize(cufftWorkSize);
      auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());

      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
    }
    template<class HydroKernel>
    BDHI2D<HydroKernel>::~BDHI2D(){
      CudaCheckError();
      CudaSafeCall(cudaDeviceSynchronize());
      CufftSafeCall(cufftDestroy(cufft_plan_inverse));
      CufftSafeCall(cufftDestroy(cufft_plan_forward));
      CudaSafeCall(cudaStreamDestroy(st));
      CudaSafeCall(cudaStreamDestroy(st2));
      CudaCheckError();
    }
    namespace BDHI2D_ns{

      using cufftComplex2 = BDHI2D_ns::cufftComplex2;
      using cufftComplex = BDHI2D_ns::cufftComplex;
      using cufftReal = BDHI2D_ns::cufftReal;

      /*This function takes a node index and returns the corresponding wave number*/
      template<class vec2>
      inline __device__ vec2 cellToWaveNumber(const int2 &cell, const int2 &cellDim, const vec2 &L){
	const vec2 pi2invL = (real(2.0)*real(M_PI))/L;
	/*My wave number*/
	vec2 k = {cell.x*pi2invL.x,
		  cell.y*pi2invL.y};
	/*Be careful with the conjugates*/
	/*Remember that FFT stores wave numbers as K=0:N/2+1:-N/2:-1 */
	if(cell.x >= (cellDim.x+1)/2) k.x -= real(cellDim.x)*pi2invL.x;
	if(cell.y >= (cellDim.y+1)/2) k.y -= real(cellDim.y)*pi2invL.y;
	return k;
      }

      //Performs the product G_k*factor_k, given f_k and g_k, see Algorithm 1.3 in [1]
      inline __device__ cufftComplex2 projectFourier(const real2 &k, const cufftComplex2 &factor,
						     real fk, real gk){

	cufftComplex2 res;
	{//Real part
	  const real2 fr = make_real2(factor.x.x, factor.y.x);
	  const real2 vr = { k.y*gk*dot(fr, {k.y, -k.x}) + k.x*fk*dot(fr, k),
			     k.x*gk*dot(fr, {-k.y, k.x}) + k.y*fk*dot(fr, k)};
	  res.x.x = vr.x;
	  res.y.x = vr.y;
	}
	{//Imaginary part
	  const real2 fi = make_real2(factor.x.y, factor.y.y);
	  const real2 vi = { k.y*gk*dot(fi, {k.y, -k.x}) + k.x*fk*dot(fi, k),
			     k.x*gk*dot(fi, {-k.y, k.x}) + k.y*fk*dot(fi, k)};
	  res.x.y = vi.x;
	  res.y.y = vi.y;
	}
	return res;
      }

      template<class HydroKernel>
      __global__ void forceFourier2Vel(cufftComplex2 * gridForces, /*Input array*/
				       cufftComplex2 * gridVels, /*Output array, can be the same as input*/
				       real vis,
				       Grid grid,/*Grid information and methods*/
				       HydroKernel hydroKernel,
				       real hydrodynamicRadius
				       ){
	/*Get my cell*/
	int2 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;


	if(cell.x>=grid.cellDim.x/2+2) return;
	if(cell.y>=grid.cellDim.y) return;

	const int icell = grid.getCellIndex(cell);
	if(icell == 0){
	  gridVels[0] = cufftComplex2();
	  return;
	}
	const int ncells = grid.getNumberCells();
	const real2 k = cellToWaveNumber(cell, make_int2(grid.cellDim), make_real2(grid.box.boxSize));
	const real k2 = dot(k,k);

	const real a = hydrodynamicRadius;
	const real2 fg = hydroKernel(k2,a);
	const real fk = fg.x/(vis*real(ncells));
	const real gk = fg.y/(vis*real(ncells));
	const cufftComplex2 factor = gridForces[icell];
	gridVels[icell] = projectFourier(k, factor, fk, gk);
      }

      template<class HydroKernel>
      __global__ void fourierBrownianNoise(/*Values of vels on each cell*/
					   cufftComplex2 *__restrict__ gridVelsFourier,
					   Grid grid, /*Grid parameters. Size of a cell, number of cells...*/
					   real prefactor,
					   real vis,
					   //Parameters to seed the RNG
					   ullint seed,
					   ullint step,
					   HydroKernel hydroKernel,
					   real hydrodynamicRadius
					   
					   ){
	int2 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	const int icell = grid.getCellIndex(cell);
	if(2*cell.x >= grid.cellDim.x+1 or cell.y >= grid.cellDim.y) return;

	if(cell.x == 0 and cell.y == 0) return;
	if(cell.x==0 and 2*cell.y >= grid.cellDim.y+1) return;

	const bool isXnyquist = (cell.x == grid.cellDim.x - cell.x) && (grid.cellDim.x%2 == 0);
	const bool isYnyquist = (cell.y == grid.cellDim.y - cell.y) && (grid.cellDim.y%2 == 0);

	  //if(isYnyquist or isXnyquist) return;


	cufftComplex2 noise;
	{
	  Saru saru(icell, step, seed);
	  const real complex_gaussian_sc = real(0.707106781186547)*prefactor; //1/sqrt(2)
	  noise.x =make_real2(saru.gf(0, complex_gaussian_sc));
	  noise.y =make_real2(saru.gf(0, complex_gaussian_sc));
	  if((isXnyquist and cell.y == 0) or
	     (isYnyquist and cell.x == 0) or
	     (isXnyquist and isYnyquist )){
	    noise.x.x *= real(1.41421356237310);
	    noise.y.x *= real(1.41421356237310);
	    noise.x.y = 0;
	    noise.y.y = 0;
	  }
	}

	{
	  const real2 k = cellToWaveNumber(cell, make_int2(grid.cellDim), make_real2(grid.box.boxSize));
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
	  gridVelsFourier[icell] += factor;
	  //Ensure correct conjugacy v_i = \conj{v_{N-i}}
	  if(cell.x == 0 and cell.y < grid.cellDim.y/2){
	    factor.x.y *= real(-1.0);
	    factor.y.y *= real(-1.0);
	    gridVelsFourier[cell.x+(grid.cellDim.y - cell.y)*grid.cellDim.x] += factor;
	  }

	}

      }

      struct toReal2{
	template<class vtype>
	inline __device__ real2 operator()(vtype q){ return make_real2(q);}
      };
    }

    template<class HydroKernel>
    void BDHI2D<HydroKernel>::spreadParticles(){

      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      real2* d_gridVels = (real2*)thrust::raw_pointer_cast(gridVelsFourier.data());

      //Spread thermal drift only if needed
      if(hydroKernel->hasThermalDrift() and temperature > 0){
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Spreading thermal drift");

	real variance = hydroKernel->getGaussianVariance(hydrodynamicRadius);
	auto tr = thrust::make_constant_iterator<real>(temperature/variance);

	ibmThermalDrift->spread(pos.begin(), tr, d_gridVels, numberParticles, st2);
      }

      //Spread forces and thermal drift only when needed
      if(interactors.size()>0){
	auto force = pd->getForce(access::location::gpu, access::mode::read);
	auto f_tr = thrust::make_transform_iterator(force.begin(), BDHI2D_ns::toReal2());
	  sys->log<System::DEBUG2>("[BDHI::BDHI2D] Spread particle forces");
	  ibm->spread(pos.begin(), f_tr, d_gridVels, numberParticles, st);
      }
      CudaSafeCall(cudaStreamSynchronize(st2));
      CudaCheckError();
    }
    template<class HydroKernel>
    void BDHI2D<HydroKernel>::convolveFourier(){
      CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
      CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));

      auto d_gridVels = thrust::raw_pointer_cast(gridVelsFourier.data());
      auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());

      dim3 NthreadsCells = dim3(16,16 ,1);
      dim3 NblocksCells;
      {
	int ncellsx = grid.cellDim.x/2+1;
	NblocksCells.x= (ncellsx/NthreadsCells.x + ((ncellsx%NthreadsCells.x)?1:0));
	NblocksCells.y= grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
	NblocksCells.z = 1;
     }

      if(interactors.size() > 0 or
	 (hydroKernel->hasThermalDrift() and temperature > 0)){
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Taking grid to wave space");
	{
	  /*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
	  auto cufftStatus =
	    cufftExecReal2Complex<real>(cufft_plan_forward,
					(cufftReal*)d_gridVels,
					(cufftComplex*)d_gridVelsFourier);
	  if(cufftStatus != CUFFT_SUCCESS){
	    sys->log<System::CRITICAL>("[BDHI::BDHI2D] Error in forward CUFFT");
	  }
	}
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Wave space convolution");
	BDHI2D_ns::forceFourier2Vel<<<NblocksCells, NthreadsCells, 0, st>>>
	  ((cufftComplex2*) d_gridVelsFourier, //Input: FFT·S·F
	   (cufftComplex2*) d_gridVelsFourier, //Output: B·FFT·S·F
	   viscosity,
	   grid,
	   *hydroKernel,
	   hydrodynamicRadius);
      }
      /*Add the stochastic term to the fourier velocities if T>0 -> 1/√σ·√B·dWw */
      if(temperature > 0){
	static ullint counter = 0; //Seed the rng differently each call
	counter++;
	sys->log<System::DEBUG2>("[BDHI::BDHI2D] Wave space brownian noise");
	//real prefactor = sqrt(2.0*temperature/(dt*grid.getCellVolume()));
	real prefactor = sqrt(2.0*temperature/(viscosity*dt*box.boxSize.x*box.boxSize.y));
	BDHI2D_ns::fourierBrownianNoise<<<NblocksCells, NthreadsCells, 0, st>>>(
										//In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw
										(cufftComplex2*)d_gridVelsFourier,
										grid,
										prefactor,
										viscosity,
										seed, //Saru needs two seeds apart from thread id
										counter,
										*hydroKernel,
										hydrodynamicRadius);

      }

      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Going back to real space");
      {
	/*Take the fourier velocities back to real space ->  FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
	auto cufftStatus =
	  cufftExecComplex2Real<real>(cufft_plan_inverse,
		       (cufftComplex*)d_gridVelsFourier,
		       (cufftReal*)d_gridVels);
	if(cufftStatus != CUFFT_SUCCESS){
	  sys->log<System::CRITICAL>("[BDHI::BDHI2D] Error in inverse CUFFT");
	}
      }
      CudaCheckError();

    }
    template<class HydroKernel>
    void BDHI2D<HydroKernel>::interpolateParticles(){
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Interpolate grid velocities to particles");
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      real2* d_gridVels = (real2*)thrust::raw_pointer_cast(gridVelsFourier.data());
      real2* d_particleVels = thrust::raw_pointer_cast(particleVels.data());
      ibm->gather(pos,
		  d_particleVels,
		  d_gridVels,
		  numberParticles, st);
    }

    namespace BDHI2D_ns{
      struct euler_functor: public thrust::binary_function<real2, real4, real4>{
	real dt;
	euler_functor(real _dt) : dt(_dt) {}
	__host__ __device__
        real4 operator()(const real2& vel, const real4& pos) const {
	  return pos + make_real4(vel*dt);
        }
      };

    }
    template<class HydroKernel>
    void BDHI2D<HydroKernel>::forwardTime(){
      sys->log<System::DEBUG1>("[BDHI::BDHI2D] Performing step");
      sys->log<System::DEBUG1>("[BDHI::BDHI2D] Computing particle displacements....");
      //Clean
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Setting vels to zero...");
      thrust::fill(thrust::cuda::par.on(st2), particleVels.begin(), particleVels.end(), real2());
      thrust::fill(gridVelsFourier.begin(), gridVelsFourier.end(),
		   typename decltype(gridVelsFourier)::value_type());

      spreadParticles();      
      convolveFourier();      
      interpolateParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      const int numberParticles = pg->getNumberParticles();
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Updating positions");
      try{
	thrust::transform(thrust::cuda::par,
			  particleVels.begin(), particleVels.end(),
			  pos.begin(),
			  pos.begin(),
			  BDHI2D_ns::euler_functor(dt));
      }
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[BDHI::BDHI2D] Thrust transform failed with: %s", e.what());
      }
      sys->log<System::DEBUG2>("[BDHI::BDHI2D] Done");
      cudaDeviceSynchronize();
      CudaCheckError();
    }

  }
}

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
namespace uammd{
  namespace BDHI{

    Quasi2D::Quasi2D(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys,
	     Parameters par):
      Integrator(pd, pg, sys, "BDHI::Quasi2D"),
      dt(par.dt),
      temperature(par.temperature),
      viscosity(par.viscosity),      
      box(par.box), grid(par.box, int3()){

      seed = sys->rng().next();
      sys->log<System::MESSAGE>("[BDHI::Quasi2D] Initialized");
      sys->log<System::MESSAGE>("[BDHI::Quasi2D] Using kernel: %s", type_name<Kernel>().c_str());
      if(box.boxSize.x == real(0.0) && box.boxSize.y == real(0.0)){
	sys->log<System::CRITICAL>("[BDHI::Quasi2D] Box of size zero detected, cannot work without a box! (make sure a box parameter was passed)");
      }
      box = Box(make_real3(box.boxSize.x, box.boxSize.y, 0));

      int numberParticles = pg->getNumberParticles();
	
      int3 cellDim;
      if(par.cells.x<=0){
	if(par.hydrodynamicRadius<=0)
	  sys->log<System::CRITICAL>("[BDHI::Quasi2D] I need an hydrodynamic radius if cell dimensions are not provided!");
	real h = par.hydrodynamicRadius;
	cellDim = nextFFTWiseSize3D(make_int3(box.boxSize/h));
      }
      else{
        cellDim = make_int3(par.cells,1);
      }
      cellDim.z = 1;
      grid = Grid(box, cellDim);

      int support  = (int(3.0 * par.hydrodynamicRadius * grid.cellDim.x / grid.box.boxSize.x) +1 )*2+1;
      
      double width = pow(par.hydrodynamicRadius * 0.66556976637237890625, 2);
      auto kernel = std::make_shared<Kernel>(support, width);
      ibm = std::make_shared<IBM<Kernel>>(sys, kernel);
      
      
      //Try to set the closest rh possible
      if(par.cells.x<=0){
       	double h = grid.cellSize.x;
       	cellDim = nextFFTWiseSize3D(make_int3(box.boxSize/h));
	cellDim.z = 1;
       	grid = Grid(box, cellDim);
	kernel = std::make_shared<Kernel>(support, width);
	ibm = std::make_shared<IBM<Kernel>>(sys, kernel);
      }
      double rh = par.hydrodynamicRadius;
      
      if(box.boxSize.x != box.boxSize.y){
	sys->log<System::WARNING>("[BDHI::Quasi2D] Self mobility will be different for non square boxes!");
      }

      sys->log<System::MESSAGE>("[BDHI::Quasi2D] Box Size: %g %g", grid.box.boxSize.x, grid.box.boxSize.y);

      sys->log<System::MESSAGE>("[BDHI::Quasi2D] Grid dimensions: %d %d", grid.cellDim.x, grid.cellDim.y);
      sys->log<System::MESSAGE>("[BDHI::Quasi2D] Interpolation kernel support: %g rh max distance, %d cells total", kernel->support*0.5*grid.cellSize.x/rh, kernel->support);
      sys->log<System::MESSAGE>("[BDHI::Quasi2D] Gaussian variance: %g", width);

      sys->log<System::MESSAGE>("[BDHI::Quasi2D] h: %g %g", grid.cellSize.x, grid.cellSize.y);
      sys->log<System::MESSAGE>("[BDHI::Quasi2D] Cell volume: %e", grid.cellSize.x*grid.cellSize.y);
      sys->log<System::MESSAGE>("[BDHI::Quasi2D] Requested kernel tolerance: %g", par.tolerance);

      if(kernel->support >= grid.cellDim.x or
	 kernel->support >= grid.cellDim.y)
	sys->log<System::CRITICAL>("[BDHI::Quasi2D] Kernel support is too big, try lowering the tolerance or increasing the box size!.");
      
      CudaSafeCall(cudaStreamCreate(&st));

  
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

    void Quasi2D::initCuFFT(){
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

      sys->log<System::DEBUG>("[BDHI::Quasi2D] cuFFT grid size: %d %d", cdtmp.x, cdtmp.y);
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
      
      sys->log<System::DEBUG>("[BDHI::Quasi2D] Necessary work space for cuFFT: %s, available: %s, total: %s",
			      printUtils::prettySize(cufftWorkSize).c_str(),
			      printUtils::prettySize(free_mem).c_str(),
			      printUtils::prettySize(total_mem).c_str());

      if(free_mem<cufftWorkSize){
	sys->log<System::CRITICAL>("[BDHI::Quasi2D] Not enough memory in device to allocate cuFFT free %s, needed: %s!!",
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(cufftWorkSize).c_str());
      }

      cufftWorkArea.resize(cufftWorkSize);
      auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      
      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
    }

    Quasi2D::~Quasi2D(){
      CudaCheckError();
      CudaSafeCall(cudaDeviceSynchronize());
      CufftSafeCall(cufftDestroy(cufft_plan_inverse));
      CufftSafeCall(cufftDestroy(cufft_plan_forward));
      CudaSafeCall(cudaStreamDestroy(st));
      CudaCheckError();
    }
    namespace Quasi2D_ns{
      
      using cufftComplex2 = Quasi2D::cufftComplex2;
      using cufftComplex = Quasi2D::cufftComplex;
      using cufftReal = Quasi2D::cufftReal;

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

      inline __device__ cufftComplex2 projectFourier(const real2 &k, const cufftComplex2 &factor,
						     real fk, real gk){

	cufftComplex2 res;
	const real k2 = dot(k, k);
	{//Real part
	  const real2 fr = make_real2(factor.x.x, factor.y.x);
	  const real2 vr = {fr.x*k2*(gk+fk) + fr.y*k.x*k.y*(fk-gk),
			    fr.x*k.x*k.y*(fk-gk) + fr.y*k2*(fk+gk)};
	  res.x.x = vr.x;
	  res.y.x = vr.y;
	}
	{//Imaginary part
	  const real2 fi = make_real2(factor.x.y, factor.y.y);
	  const real2 vi = {fi.x*k2*(gk+fk) + fi.y*k.x*k.y*(fk-gk),
			    fi.x*k.x*k.y*(fk-gk) + fi.y*k2*(fk+gk)};
	  res.x.y = vi.x;
	  res.y.y = vi.y;
	}
	return res;
      }
      
      __global__ void forceFourier2Vel(cufftComplex2 * gridForces, /*Input array*/
				       cufftComplex2 * gridVels, /*Output array, can be the same as input*/
				       real vis,				       
				       Grid grid/*Grid information and methods*/
				       ){
	/*Get my cell*/
	int2 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;


	if(cell.x>=grid.cellDim.x/2+2) return;
	if(cell.y>=grid.cellDim.y) return;
	
	const int icell = grid.getCellIndex(cell);
	if(icell == 0){
	  gridVels[0] = {0,0, 0,0};
	  return;
	}
	const int ncells = grid.getNumberCells();
	const real2 k = cellToWaveNumber(cell, make_int2(grid.cellDim), make_real2(grid.box.boxSize));
	const real k2 = dot(k,k);


	const cufftComplex2 factor = gridForces[icell];
	
	const real fk = 0;
	const real gk = real(1.0)/(k2*k2*vis*real(ncells));
	gridVels[icell] = projectFourier(k, factor, fk, gk);
      }

      __global__ void fourierBrownianNoise(/*Values of vels on each cell*/
					   cufftComplex2 *__restrict__ gridVelsFourier, 
					   Grid grid, /*Grid parameters. Size of a cell, number of cells...*/
					   real prefactor,
					   real vis,	      
					   //Parameters to seed the RNG					   
					   ullint seed,
					   ullint step
					   ){
	int2 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	const int icell = grid.getCellIndex(cell);
	if(2*cell.x >= grid.cellDim.x+1 or cell.y >= grid.cellDim.y) return;	
	
	if(cell.x == 0 and cell.y == 0) return;
	if(cell.x==0 and 2*cell.y >= grid.cellDim.y+1) return;
	{
	  const bool isXnyquist = (cell.x == grid.cellDim.x - cell.x) && (grid.cellDim.x%2 == 0);
	  const bool isYnyquist = (cell.y == grid.cellDim.y - cell.y) && (grid.cellDim.y%2 == 0);

	  if(isYnyquist or isXnyquist) return;
	}
	
	cufftComplex2 noise;
	{
	  Saru saru(icell, step, seed);
	  const real complex_gaussian_sc = real(0.707106781186547)*prefactor; //1/sqrt(2)
	  noise.x =make_real2(saru.gf(0, complex_gaussian_sc));
	  noise.y =make_real2(saru.gf(0, complex_gaussian_sc));	  
	}

	{
	  const real2 k = cellToWaveNumber(cell, make_int2(grid.cellDim), make_real2(grid.box.boxSize));
	  cufftComplex2 factor;
	  {
	    const real k2 = dot(k,k);
	    
	    const real fk = 0;
	    const real gk = real(1.0)/(k2*k2);

	    const real fk_sq = sqrt(fk/(vis*grid.getNumberCells()));
	    const real gk_sq = sqrt(gk/(vis*grid.getNumberCells()));

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
    
    void Quasi2D::spreadParticles(){
      sys->log<System::DEBUG2>("[BDHI::Quasi2D] Spread particle forces");
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      real2* d_gridVels = (real2*)thrust::raw_pointer_cast(gridVelsFourier.data());
            
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      auto tr = thrust::make_transform_iterator(force.begin(), Quasi2D_ns::toReal2());
      
      ibm->spread(pos.begin(), tr, d_gridVels, grid, numberParticles, st);
    }
    void Quasi2D::convolveFourier(){
      CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
      CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));

      auto d_gridVels = thrust::raw_pointer_cast(gridVelsFourier.data());
      auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());

      if(interactors.size() > 0){
	sys->log<System::DEBUG2>("[BDHI::Quasi2D] Taking grid to wave space");
	{
	  /*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
	  auto cufftStatus =
	    cufftExecReal2Complex<real>(cufft_plan_forward,
					(cufftReal*)d_gridVels,
					(cufftComplex*)d_gridVelsFourier);
	  if(cufftStatus != CUFFT_SUCCESS){
	    sys->log<System::CRITICAL>("[BDHI::Quasi2D] Error in forward CUFFT");
	  }
	}
      }
      {      
	dim3 NthreadsCells = dim3(16,16 ,1);
	dim3 NblocksCells;
	{
	  int ncellsx = grid.cellDim.x/2+1;
	  NblocksCells.x= (ncellsx/NthreadsCells.x + ((ncellsx%NthreadsCells.x)?1:0));
	  NblocksCells.y= grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
	  NblocksCells.z = 1;
	}	
	if(interactors.size() > 0){
	  sys->log<System::DEBUG2>("[BDHI::Quasi2D] Wave space convolution");
	  Quasi2D_ns::forceFourier2Vel<<<NblocksCells, NthreadsCells, 0, st>>>
	    ((cufftComplex2*) d_gridVelsFourier, //Input: FFT·S·F
	     (cufftComplex2*) d_gridVelsFourier, //Output: B·FFT·S·F
	     viscosity,
	     grid);
	}
	/*Add the stochastic term to the fourier velocities if T>0 -> 1/√σ·√B·dWw */
	if(temperature > real(0.0)){
	  static ullint counter = 0; //Seed the rng differently each call
	  counter++;
	  sys->log<System::DEBUG2>("[BDHI::Quasi2D] Wave space brownian noise");
	  real prefactor = sqrt(2.0*temperature/(dt*grid.getCellVolume()));
	  Quasi2D_ns::fourierBrownianNoise<<<NblocksCells, NthreadsCells, 0, st>>>(
			//In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw 
			(cufftComplex2*)d_gridVelsFourier, 
			grid,
			prefactor,
			viscosity,
			seed, //Saru needs two seeds apart from thread id
			counter);
	}
      }

      sys->log<System::DEBUG2>("[BDHI::Quasi2D] Going back to real space");
      {
	/*Take the fourier velocities back to real space ->  FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
	auto cufftStatus =
	  cufftExecComplex2Real<real>(cufft_plan_inverse,
		       (cufftComplex*)d_gridVelsFourier,
		       (cufftReal*)d_gridVels);
	if(cufftStatus != CUFFT_SUCCESS){
	  sys->log<System::CRITICAL>("[BDHI::Quasi2D] Error in inverse CUFFT");
	}
      }
      CudaCheckError();

    }

    void Quasi2D::interpolateParticles(){
      sys->log<System::DEBUG2>("[BDHI::Quasi2D] Interpolate grid velocities to particles");	    
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      real2* d_gridVels = (real2*)thrust::raw_pointer_cast(gridVelsFourier.data());
      real2* d_particleVels = thrust::raw_pointer_cast(particleVels.data());
      
      ibm->gather(pos,
		  d_particleVels,
		  d_gridVels,
		  grid, numberParticles, st);      
    }

    void Quasi2D::computeMF(){
      sys->log<System::DEBUG1>("[BDHI::Quasi2D] Computing particle displacements....");      
      /*Clean gridVels*/
      {
	sys->log<System::DEBUG2>("[BDHI::Quasi2D] Setting vels to zero...");
	thrust::fill(gridVelsFourier.begin(), gridVelsFourier.end(), decltype(gridVelsFourier)::value_type());
      }
      if(interactors.size()>0) spreadParticles();
      convolveFourier();
      
      thrust::fill(particleVels.begin(), particleVels.end(), real2());
      interpolateParticles();      
      
    }

    namespace Quasi2D_ns{
      struct euler_functor: public thrust::binary_function<real2, real4, real4>{
	real dt;
	euler_functor(real _dt) : dt(_dt) {}
	__host__ __device__
        real4 operator()(const real2& vel, const real4& pos) const { 
	  return pos + make_real4(vel*dt);
        }
      };
      
    }
    
    void Quasi2D::forwardTime(){
      sys->log<System::DEBUG1>("[BDHI::Quasi2D] Performing step");
      computeMF();
      cudaDeviceSynchronize();
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      sys->log<System::DEBUG2>("[BDHI::Quasi2D] Updating positions");
      try{
	thrust::transform(thrust::device,
			  particleVels.begin(), particleVels.end(),
			  pos.begin(),
			  pos.begin(),
			  Quasi2D_ns::euler_functor(dt));
      }
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[BDHI::Quasi2D] Thrust transform failed with: %s", e.what());
      }
      sys->log<System::DEBUG2>("[BDHI::Quasi2D] Done");
    }

  }
}

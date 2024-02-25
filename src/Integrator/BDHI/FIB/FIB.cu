/*Raul P. Pelaez 2018. Fluctuating Immerse Boundary for Brownian Dynamics with Hydrodynamic Interactions.

This file implements the algorithm described in [1] for PBC using FFT to solve the stokes operator. Fluid properties are stored in a staggered grid for improved translational invariance [2]. This module solves the same regime as the rest of the BDHI modules, but without imposing a mobility kernel. FIB solves the fluctuating Navier-Stokes equation directly.

See FIB.cuh for more info.

References:
[1] Brownian Dynamics without Green's Functions. Steve Delong, Florencio Balboa, et. al. 2014
[2] Staggered Schemes for Fluctuating Hydrodynamics. Florencio Balboa, et.al. 2012.
 */
#include"FIB.cuh"
#include "global/defines.h"
#include"utils/cufftDebug.h"
#include"utils/curandDebug.h"
#include"utils/debugTools.h"
#include"third_party/saruprng.cuh"
#include"utils/cuda_lib_defines.h"
#include"utils/atomics.cuh"
namespace uammd{
  namespace BDHI{

    namespace FIB_ns{
      //Looks for the closest (equal or greater) number of nodes of the form 2^a*3^b*5^c*7^d*11^e
      int3 nextFFTWiseSize3D(int3 size){
	int* cdim = &size.x;
	int max_dim = std::max({size.x, size.y, size.z});
	int n= 10;
	int n5 = 6; //number higher than this are not reasonable...
	int n7 = 5;
	int n11 = 4;
	std::vector<int> tmp(n*n*n5*n7*n11, 0);
	do{
	  tmp.resize(n*n*n5*n7*n11, 0);
	  fori(0,n)forj(0,n)
	    for(int k=0; k<n5;k++)for(int k7=0; k7<n7; k7++)for(int k11=0; k11<n11; k11++){
		if(k11>4 or k7>5 or k>6) continue;

		int id = i+n*j+n*n*k+n*n*n5*k7+n*n*n5*n7*k11;
		tmp[id] = 0;
		//Current fft wise size
		int number = pow(2,i)*pow(3,j)*pow(5,k)*pow(7, k7)*pow(11, k11);
		//The fastest FFTs always have at least a factor of 2
		if(i==0) continue;
		//I have seen empirically that factors 11 and 7 only works well with at least a factor 2 involved
		if((k11>0 && (i==0))) continue;
		tmp[id] = number;
	      }
	  n++;
	  /*Sort this array in ascending order*/
	  std::sort(tmp.begin(), tmp.end());
	}while(tmp.back()<max_dim); /*if n is not enough, include more*/

	//I have empirically seen that these sizes produce slower FFTs than they should in several platforms
	constexpr int forbiddenSizes [] = {28, 98, 150, 154, 162, 196, 242};
	/*Now look for the nearest value in tmp that is greater than each cell dimension and it is not forbidden*/
	forj(0,3){
	  fori(0, tmp.size()){
	    if(tmp[i]<cdim[j]) continue;
	    for(int k =0;k<sizeof(forbiddenSizes)/sizeof(int); k++) if(tmp[i] == forbiddenSizes[k]) continue;
	    cdim[j] = tmp[i];
	    break;
	  }
	}
	return size;
      }
    }

    FIB::FIB(shared_ptr<ParticleGroup> pg, Parameters par):
      Integrator(pg, "BDHI::FIB"),
      dt(par.dt),
      temperature(par.temperature), viscosity(par.viscosity),
      box(par.box), scheme(par.scheme){
      CudaCheckError();
      sys->log<System::MESSAGE>("[BDHI::FIB] Initialized");
      seed = sys->rng().next();
      int numberParticles = pg->getNumberParticles();
      if(par.hydrodynamicRadius>0 and par.cells.x > 0)
	sys->log<System::CRITICAL>("[BDHI::FIB] Please provide hydrodynamic radius OR cell dimensions, not both.");
      int3 cellDim = par.cells;
      //If cells were not provided compute the closest one for the requested Rh
      if(par.cells.x<0){
	if(par.hydrodynamicRadius<0)
	  sys->log<System::CRITICAL>("[BHDI::FIB] I need either the hydrodynamic radius or the number of cells!");
	real hgrid = Kernel::adviseGridSize(par.hydrodynamicRadius, par.tolerance);
	cellDim = make_int3(box.boxSize/hgrid);
	/*FFT likes a number of cells as cellDim.i = 2^n·3^l·5^m */
	cellDim = FIB_ns::nextFFTWiseSize3D(cellDim);
      }
      if(par.cells.x>0) cellDim = par.cells;
      if(par.hydrodynamicRadius>0)
	sys->log<System::MESSAGE>("[BDHI::FIB] Target hydrodynamic radius: %f", par.hydrodynamicRadius);
      if(cellDim.x <3)cellDim.x = 3;
      if(cellDim.y <3)cellDim.y = 3;
      if(cellDim.z==2)cellDim.z = 3;
      /*Store grid parameters in a Mesh object*/
      this->grid = Grid(box, cellDim);
      real h = std::min({grid.cellSize.x, grid.cellSize.y, grid.cellSize.z});
      this->kernel = std::make_shared<Kernel>(h, par.tolerance);
      this->hydrodynamicRadius = kernel->fixHydrodynamicRadius(h, grid.cellSize.x);
      double rh = this->hydrodynamicRadius;
#ifndef SINGLE_PRECISION
      this->deltaRFD = 1e-6*rh;
#else
      this->deltaRFD = 1e-4*rh;
#endif

      /*Print information*/
      sys->log<System::MESSAGE>("[BDHI::FIB] Closest possible hydrodynamic radius: %f", rh);
      sys->log<System::MESSAGE>("[BDHI::FIB] Box Size: %f %f %f", box.boxSize.x, box.boxSize.y, box.boxSize.z);
      sys->log<System::MESSAGE>("[BDHI::FIB] Mesh dimensions: %d %d %d", cellDim.x, cellDim.y, cellDim.z);
      sys->log<System::MESSAGE>("[BDHI::FIB] Temperature: %f", temperature);
      sys->log<System::MESSAGE>("[BDHI::FIB] Self mobility: %f", this->getSelfMobility());
      if(box.boxSize.x != box.boxSize.y || box.boxSize.y != box.boxSize.z || box.boxSize.x != box.boxSize.z){
	sys->log<System::WARNING>("[BDHI::FCM] Self mobility will be different for non cubic boxes!");
      }
      sys->log<System::MESSAGE>("[BDHI::FIB] Random Finite Diference delta: %e", deltaRFD);
      sys->log<System::MESSAGE>("[BDHI::FIB] dt: %f", dt);
      if(scheme == Scheme::MIDPOINT)
	sys->log<System::MESSAGE>("[BDHI::FIB] Temporal integrator: Simple midpoint");
      else if(scheme == Scheme::IMPROVED_MIDPOINT)
	sys->log<System::MESSAGE>("[BDHI::FIB] Temporal integrator: Improved midpoint");
      int ncells = grid.getNumberCells();
      CudaCheckError();
      //Init rng
      CurandSafeCall(curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT));
      //curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_MT19937);
      CurandSafeCall(curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next()));
      try{
	thrust::device_vector<real> noise(30000);
	auto noise_ptr = thrust::raw_pointer_cast(noise.data());
	//Warm cuRNG
	CurandSafeCall(curandgeneratenormal(curng, noise_ptr, noise.size(), 0.0, 1.0));
	CurandSafeCall(curandgeneratenormal(curng, noise_ptr, noise.size(), 0.0, 1.0));
      }
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[BDHI::FIB] Thrust could not allocate necessary arrays at initialization with error: %s", e.what());
      }
      CufftSafeCall(cufftCreate(&cufft_plan_forward));
      CufftSafeCall(cufftCreate(&cufft_plan_inverse));
      /*I will be handling workspace memory*/
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));
      //Required storage for the plans
      size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
      /*Set up cuFFT*/
      int3 cdtmp = {grid.cellDim.z, grid.cellDim.y, grid.cellDim.x};
      /*I want to make three 3D FFTs, each one using one of the three interleaved coordinates*/
      CufftSafeCall(cufftMakePlanMany(cufft_plan_forward,
				      3, &cdtmp.x, /*Three dimensional FFT*/
				      &cdtmp.x,
				      /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
				      3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
				      /*Same format in the output*/
				      &cdtmp.x,
				      3, 1,
				      /*Perform 3 direct Batched FFTs*/
				      CUFFT_Real2Complex<real>::value, 3,
				      &cufftWorkSizef));
      sys->log<System::DEBUG>("[BDHI::FIB] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
      /*Same as above, but with C2R for inverse FFT*/
      CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
				      3, &cdtmp.x, /*Three dimensional FFT*/
				      &cdtmp.x,
				      /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
				      3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
				      &cdtmp.x,
				      3, 1,
				      /*Perform 3 inverse batched FFTs*/
				      CUFFT_Complex2Real<real>::value, 3,
				      &cufftWorkSizei));
      /*Allocate cuFFT work area*/
      size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei);
      sys->log<System::DEBUG>("[BDHI::FIB] Necessary work space for cuFFT: %s", printUtils::prettySize(cufftWorkSize).c_str());
      size_t free_mem, total_mem;
      CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
      if(free_mem<cufftWorkSize){
	sys->log<System::CRITICAL>("[BDHI::FIB] Not enough memory in device to allocate cuFFT free %s, needed: %s!!!",
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(cufftWorkSize).c_str());
      }
      try{
	cufftWorkArea.resize(cufftWorkSize);
      }
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[BDHI::FIB] Thrust could not allocate cufft work area of size: %s (free memory: %s), with error: %s",
				   printUtils::prettySize(cufftWorkSize).c_str(),
				   printUtils::prettySize(free_mem).c_str(),
				   e.what());
      }
      try{
	if(temperature > real(0.0)){
	  sys->log<System::DEBUG>("[BDHI::FIB] Allocating random");
	  random.resize(6*ncells);
	}
	//Grid velocities, same array is used for real and Fourier space
	/*The layout of this array is
	  fx000, fy000, fz000, fx001, fy001, fz001..., fxnnn, fynnn, fznnn. n=ncells-1
	  When used in real space each f is a real number, whereas in wave space each f will be a complex number.
	*/
	sys->log<System::DEBUG>("[BDHI::FIB] Allocating gridVels");
	gridVelsFourier.resize(3*ncells, cufftComplex());
	gridVels.resize(ncells, real3());
	sys->log<System::DEBUG>("[BDHI::FIB] Allocating posOld");
	posOld.resize(numberParticles);
      }
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[BDHI::FIB] Thrust could not allocate necessary arrays at initialization with error: %s", e.what());
      }
      void * d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, d_cufftWorkArea));
      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
    }
    FIB::~FIB(){
      cudaDeviceSynchronize();
      cufftDestroy(cufft_plan_inverse);
      cufftDestroy(cufft_plan_forward);
      curandDestroyGenerator(curng);
      sys->log<System::DEBUG>("[BDHI::FIB] Destroyed");
    }

    namespace FIB_ns{
      struct Direction{
	static constexpr int XX = 0, YY=1, ZZ=2, XY=3, XZ=4, YZ=5;
      };

      //Adds the stochastic term to the current velocity: v_i += noisePrefactor*\hat{D}·W^n
      //noisePrefactor will be proportional to sqrt(\eta·kT/(dt·Vcell)) depending on the temporal integrator.
      //W is a symmetric tensor with 3 white gaussian numbers per direction
      //(staggered grid, W is defined at the cell centers and edges and \hat{D}·W in the faces)
      //W^{\alpha\alpha} are in the centers with variance 2
      //W^{\alpha\beta} are in the edges with variance 1
      __global__ void addRandomAdvection(real3* gridVels, //Real space velocities of each cell, defined at the cell faces
					 Grid grid,
					 real noisePrefactor,
					  //6 random numbers per cell, sorted by direction [first number for all cells, ..., second number for all cells,...]
					 const real* random){
	/*I expect a 3D grid of threads one for each grid cell*/
	/*Get my cell*/
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;
	if(cell.x>=grid.cellDim.x) return;
	if(cell.y>=grid.cellDim.y) return;
	if(cell.z>=grid.cellDim.z) return;
	/*Get my cell index (position in the array) */
	const int icell = grid.getCellIndex(cell);
	const int ncells = grid.getNumberCells();
	//I will draw 6 random numbers for each cell
	real3 DW = make_real3(0);
	//(\nabla·W)_\alpha^i = \nabla^i·(W_{\alpha x}, W_{\alpha y}, W_{\alpha z}) ->
	// -> \partial_\alpha^i W_{\alpha\beta} = 1/d\alpha (W^{\alpha\beta}_{i+\alpha/2} - W^{\alpha\beta}_{i-\alpha/2})
	constexpr real sqrt2 = real(1.41421356237310);
	//Diagonal terms
	//See eq. 10 in [1] to understand the sqrt2
	//m -> \alpha - 1/2, p -> \alpha + 1/2
	{
	  const int n_mx = icell; //n_[m/p]\alpha -> cell index of neighbour in a certain direction
	  real wxx_mx = random[n_mx+ncells*Direction::XX];
	  const int n_px = grid.getCellIndex(grid.pbc_cell({cell.x + 1, cell.y, cell.z}));
	  real wxx_px = random[n_px+ncells*Direction::XX];
	  DW.x += sqrt2*grid.invCellSize.x*(wxx_px - wxx_mx);
	}
	{
	  const int n_my = icell;
	  real wyy_my = random[n_my+ncells*Direction::YY];
	  const int n_py = grid.getCellIndex(grid.pbc_cell({cell.x, cell.y + 1, cell.z}));
	  real wyy_py = random[n_py+ncells*Direction::YY];
	  DW.y += sqrt2*grid.invCellSize.y*(wyy_py - wyy_my);
	}
	{
	  const int n_mz = icell;
	  real wzz_mz = random[n_mz+ncells*Direction::ZZ];
	  const int n_pz = grid.getCellIndex(grid.pbc_cell({cell.x, cell.y, cell.z + 1}));
	  real wzz_pz = random[n_pz+ncells*Direction::ZZ];
	  DW.z += sqrt2*grid.invCellSize.z*(wzz_pz - wzz_mz);
	}
	//Cross terms
	//W is simmetric so wxy_mx = wyx_my, etc
	{
	  const int n_m = icell;
	  real wxy_m = random[n_m + ncells*Direction::XY];
	  {
	    const int n_my = grid.getCellIndex(grid.pbc_cell({cell.x, cell.y - 1, cell.z}));
	    real wxy_my = random[n_my + ncells*Direction::XY];
	    const real dwxy_dy = (wxy_m - wxy_my);
	    DW.x += grid.invCellSize.y*dwxy_dy;
	  }
	  {
	    const int n_mx = grid.getCellIndex(grid.pbc_cell({cell.x - 1, cell.y, cell.z}));
	    real wxy_mx = random[n_mx + ncells*Direction::XY];
	    const real dwxy_dx = (wxy_m - wxy_mx);
	    DW.y += grid.invCellSize.x*dwxy_dx;
	  }
	}
	{
	  const int n_m = icell;
	  real wxz_m = random[n_m+ncells*Direction::XZ];
	  {
	    const int n_mz = grid.getCellIndex(grid.pbc_cell({cell.x, cell.y, cell.z - 1}));
	    real wxz_mz = random[n_mz+ncells*Direction::XZ];
	    const real dwxz_dz = (wxz_m - wxz_mz);
	    DW.x += grid.invCellSize.z*dwxz_dz;
	  }
	  {
	    const int n_mx = grid.getCellIndex(grid.pbc_cell({cell.x - 1, cell.y, cell.z}));
	    real wxz_mx = random[n_mx+ncells*Direction::XZ];
	    const real dwxz_dx = (wxz_m - wxz_mx);
	    DW.z += grid.invCellSize.x*dwxz_dx;
	  }
	}
	{
	  const int n_m = icell;
	  real wyz_m = random[n_m+ncells*Direction::YZ];
	  {
	    const int n_mz = grid.getCellIndex(grid.pbc_cell({cell.x, cell.y, cell.z - 1}));
	    real wyz_mz = random[n_mz+ncells*Direction::YZ];
	    const real dwyz_dz = (wyz_m - wyz_mz);
	    DW.y += grid.invCellSize.z*dwyz_dz;
	  }
	  {
	    const int n_my = grid.getCellIndex(grid.pbc_cell({cell.x, cell.y - 1, cell.z}));
	    real wyz_my = random[n_my+ncells*Direction::YZ];
	    const real dwyz_dy = (wyz_m - wyz_my);
	    DW.z += grid.invCellSize.y*dwyz_dy;
	  }
	}
	//const real dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
	//gridVels[icell] += (DW/dV)*noisePrefactor;
	gridVels[icell] += DW*noisePrefactor;
      }

      //Computes thermal drift term using RFD
      //kbT/\delta [ S(q^n + \delta/2\hat{W}^n) - S(q^n - \delta/2\hat{W}^n) ] ·\hat{W}^n
      //See eq. 32 and 33 in [1]
      template<class Kernel = IBM_kernels::Peskin::threePoint>
      __global__ void addThermalDrift(real4 *pos,
				      real3* gridVels,
				      Grid grid,
				      real driftPrefactor, //kbT/deltaRDF
				      real deltaRFD,
				      int numberParticles,
				      Kernel kernel,
				      uint seed, uint step){
		return;
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=numberParticles) return;
	const real3 pi = make_real3(pos[id]);
	//Corresponding cell of each direction in the staggered grid
	const int3 cellix = grid.getCell(make_real3(pi.x - real(0.5)*grid.cellSize.x, pi.y, pi.z));
	const int3 celliy = grid.getCell(make_real3(pi.x, pi.y - real(0.5)*grid.cellSize.y, pi.z));
	const int3 celliz = grid.getCell(make_real3(pi.x, pi.y, pi.z - real(0.5)*grid.cellSize.z));
	const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize); //Cell index to cell center position
	constexpr int P = Kernel::support/2;
	constexpr int supportCells = Kernel::support;
	constexpr int numberNeighbourCells = supportCells*supportCells*supportCells;
	real3 W;
	{
	  Saru rng(id, seed, step);
	  W = make_real3(rng.gf(real(0.0), real(1.0)),
			 rng.gf(real(0.0), real(1.0)).x);
	}
	const real3 q_p_noise = pi + real(0.5)*deltaRFD*W;
	const real3 q_m_noise = pi - real(0.5)*deltaRFD*W;
	for(int i = 0; i<numberNeighbourCells; i++){
	  /*Thermal drift of particle i in cell j*/
	  //Contribution to vx
	  {
	    //Compute neighbour cell index
	    int3 celljx = make_int3(cellix.x + i%supportCells - P,
				    cellix.y + (i/supportCells)%supportCells - P,
				    cellix.z + i/(supportCells*supportCells) - P );
	    celljx = grid.pbc_cell(celljx);

	    const int jcellx = grid.getCellIndex(celljx);
	    real SmSdWx;
	    {
	      /*Staggered distance from q + noise to center of cell j*/
	      const real3 rijx = q_p_noise-make_real3(celljx)*grid.cellSize-cellPosOffset;
	      //Spread Wx, delta with the distance of the point where vx is defined, which is 0.5dx to the left of the center
	      const auto r = grid.box.apply_pbc({rijx.x - real(0.5)*grid.cellSize.x, rijx.y, rijx.z});
	      SmSdWx =  kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.x;
	    }
	    {
	      /*Staggered distance from q - noise to center of cell j*/
	      const real3 rijx = q_m_noise-make_real3(celljx)*grid.cellSize-cellPosOffset;
	      //Spread Wx
	      const auto r = grid.box.apply_pbc({rijx.x - real(0.5)*grid.cellSize.x, rijx.y, rijx.z});
	      SmSdWx -= kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.x;
	    }
	    atomicAdd(&gridVels[jcellx].x, SmSdWx*driftPrefactor);
	  }
	  //Contribution to vy
	  {
	    int3 celljy = make_int3(celliy.x + i%supportCells - P,
				    celliy.y + (i/supportCells)%supportCells - P,
				    celliy.z + i/(supportCells*supportCells) - P );
	    celljy = grid.pbc_cell(celljy);
	    const int jcelly = grid.getCellIndex(celljy);
	    real SmSdWy;
	    {
	      const real3 rijy = q_p_noise-make_real3(celljy)*grid.cellSize-cellPosOffset;
	      const auto r = grid.box.apply_pbc({rijy.x, rijy.y - real(0.5)*grid.cellSize.y, rijy.z});
	      SmSdWy = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.y;
	    }
	    {
	      const real3 rijy = q_m_noise-make_real3(celljy)*grid.cellSize-cellPosOffset;
	      const auto r = grid.box.apply_pbc({rijy.x, rijy.y - real(0.5)*grid.cellSize.y, rijy.z});
	      SmSdWy -= kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.y;
	    }
	    atomicAdd(&gridVels[jcelly].y, SmSdWy*driftPrefactor);
	  }
	  //Contribution to vz
	  {
	    int3 celljz = make_int3(celliz.x + i%supportCells - P,
				    celliz.y + (i/supportCells)%supportCells - P,
				    celliz.z + i/(supportCells*supportCells) - P );
	    celljz = grid.pbc_cell(celljz);
	    const int jcellz = grid.getCellIndex(celljz);
	    real SmSdWz;
	    {
	      const real3 rijz = q_p_noise-make_real3(celljz)*grid.cellSize-cellPosOffset;
	      const auto r = grid.box.apply_pbc({rijz.x, rijz.y, rijz.z - real(0.5)*grid.cellSize.z});
	      SmSdWz = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.z;
	    }
	    {
	      const real3 rijz = q_m_noise-make_real3(celljz)*grid.cellSize-cellPosOffset;
	      const auto r = grid.box.apply_pbc({rijz.x, rijz.y, rijz.z - real(0.5)*grid.cellSize.z});
	      SmSdWz -= kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.z;
	    }
	    atomicAdd(&gridVels[jcellz].z, SmSdWz*driftPrefactor);
	  }
	}

      }

      template<class Grid>
      __device__ int3 computeSupportShift(real3 pos, int3 celli, Grid grid, int support){
	int3 P = make_int3(support/2);
	//Kernels with even support might need an offset of one cell depending on the position of the particle inside the cell
	const bool shift = support%2==0;
	if(shift){
	  const auto invCellSize = real(1.0)/grid.getCellSize(celli);
	  const real3 pi_pbc = grid.box.apply_pbc(pos);
	  P -= make_int3((pi_pbc+grid.box.boxSize*real(0.5))*invCellSize - make_real3(celli) + real(0.5));
	}
	return P;
      }

      //Computes S·F and adds it to gridVels
      template<class Kernel = IBM_kernels::Peskin::threePoint, class IndexIterator>
      __global__ void spreadParticleForces(real4 *pos,
					   real4 *force,
					   real3* gridVels,
					   IndexIterator index,
					   Grid grid,
					   int numberParticles,
					   Kernel kernel){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid>=numberParticles) return;
	int id = index[tid];
	const real3 pi = make_real3(pos[id]);
	const real3 forcei = make_real3(force[id]);
	//Corresponding cell of each direction in the staggered grid
	const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);
	constexpr int supportCells = Kernel::support;
	const real3 pix = make_real3(pi.x - real(0.5)*grid.cellSize.x, pi.y, pi.z);
	const int3 cellix = grid.getCell(pix);
	const int3 Px = computeSupportShift(pix, cellix, grid, supportCells);
	const real3 piy =  make_real3(pi.x, pi.y - real(0.5)*grid.cellSize.y, pi.z);
	const int3 celliy = grid.getCell(piy);
	const int3 Py = computeSupportShift(piy, celliy, grid, supportCells);
	const real3 piz = make_real3(pi.x, pi.y, pi.z - real(0.5)*grid.cellSize.z);
	const int3 celliz = grid.getCell(piz);
	const int3 Pz = computeSupportShift(piz, celliz, grid, supportCells);
	constexpr int numberNeighbourCells = supportCells*supportCells*supportCells;
	for(int i = 0; i<numberNeighbourCells; i++){
	  //Contribution to vx
	  {
	    //Compute neighbour cell index
	    int3 celljx = make_int3(cellix.x + i%supportCells,
				    cellix.y + (i/supportCells)%supportCells,
				    cellix.z + i/(supportCells*supportCells)) -Px;
	    celljx = grid.pbc_cell(celljx);
	    const int jcellx = grid.getCellIndex(celljx);
	    /*Staggered distance from q - noise to center of cell j*/
	    const real3 r = grid.distanceToCellCenter(pi - make_real3(real(0.5)*grid.cellSize.x,0,0),
							celljx);
	    //Spread Wx
	    real fx = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*forcei.x;
	    atomicAdd(&gridVels[jcellx].x, fx);
	  }
	  //Contribution to vy
	  {
	    int3 celljy = make_int3(celliy.x + i%supportCells,
				    celliy.y + (i/supportCells)%supportCells,
				    celliy.z + i/(supportCells*supportCells)) - Py;
	    celljy = grid.pbc_cell(celljy);
	    const int jcelly = grid.getCellIndex(celljy);
	    const real3 r = grid.distanceToCellCenter(pi - make_real3(0, real(0.5)*grid.cellSize.y,0),
						      celljy);

	    real fy = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*forcei.y;
	    atomicAdd(&gridVels[jcelly].y, fy);
	  }
	  //Contribution to vz
	  {
	    int3 celljz = make_int3(celliz.x + i%supportCells,
				    celliz.y + (i/supportCells)%supportCells,
				    celliz.z + i/(supportCells*supportCells))-Pz;
	    celljz = grid.pbc_cell(celljz);
	    const int jcellz = grid.getCellIndex(celljz);
	    const real3 r = grid.distanceToCellCenter(pi - make_real3(0,0, real(0.5)*grid.cellSize.z),
						      celljz);
	    real fz = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*forcei.z;
	    atomicAdd(&gridVels[jcellz].z, fz);
	  }
	}

      }

      using cufftComplex3 = FIB::cufftComplex3;

      /*This function takes a node index and returns the corresponding wave number*/
      template<class vec3>
      inline __device__ vec3 cellToWaveNumber(const int3 &cell, const int3 &cellDim, const vec3 &L){
	const vec3 pi2invL = real(2.0)*real(M_PI)/L;
	/*My wave number*/
	vec3 k = {cell.x*pi2invL.x,
		  cell.y*pi2invL.y,
		  cell.z*pi2invL.z};
	/*Be careful with the conjugates*/
	/*Remember that FFT stores wave numbers as K=0:N/2+1:-N/2:-1 */
	if(cell.x >= (cellDim.x+1)/2) k.x -= real(cellDim.x)*pi2invL.x;
	if(cell.y >= (cellDim.y+1)/2) k.y -= real(cellDim.y)*pi2invL.y;
	if(cell.z >= (cellDim.z+1)/2) k.z -= real(cellDim.z)*pi2invL.z;
	return k;
      }

      /*Apply the divergence free projection operator, P, to a wave number with a certain complex factor.
	res = (I-k^k)·factor -> k is unitary
	See i.e below eq. 45 [2]. P = I-G(DG)^-1 D = I-D*(DD*)^-1 D = I-D*L^-1 D = I-k·k^T/|k|^2
      */
      inline __device__ cufftComplex3 projectFourier(const real3 &k, const cufftComplex3 &factor){
	const real invk2 = real(1.0)/dot(k,k);
	cufftComplex3 res;
	{//Real part
	  const real3 fr = make_real3(factor.x.x, factor.y.x, factor.z.x);
	  const real kfr = dot(k,fr)*invk2;
	  const real3 vr = (fr-k*kfr);
	  res.x.x = vr.x;
	  res.y.x = vr.y;
	  res.z.x = vr.z;
	}
	{//Imaginary part
	  const real3 fi = make_real3(factor.x.y, factor.y.y, factor.z.y);
	  const real kfi = dot(k,fi)*invk2;
	  const real3 vi = (fi-k*kfi);
	  res.x.y = vi.x;
	  res.y.y = vi.y;
	  res.z.y = vi.z;
	}
	return res;
      }

      //Apply a phase shift in Fourier space to a certain v(k), given cos(phase*k) and sin(phase*k)
      inline __device__ cufftComplex3 shiftVelocity(cufftComplex3 vk, const real3 &cosk, const real3 &sink){
	FIB::cufftComplex tmp;
	tmp = vk.x;
	vk.x.x = tmp.x*cosk.x - tmp.y*sink.x;
	vk.x.y = tmp.y*cosk.x + tmp.x*sink.x;
	tmp = vk.y;
	vk.y.x = tmp.x*cosk.y - tmp.y*sink.y;
	vk.y.y = tmp.y*cosk.y + tmp.x*sink.y;
	tmp = vk.z;
	vk.z.x = tmp.x*cosk.z - tmp.y*sink.z;
	vk.z.y = tmp.y*cosk.z + tmp.x*sink.z;
	return vk;
      }

      //See eq. 28 in [1].
      //Apply 1/\eta\mathcal{L}^-1 operator to SF + noise in Fourier space. PBC makes \mathcal{L}^-1 = -P·L^-1 where P = I-G(DG)^-1 D
      __global__ void solveStokesFourier(const cufftComplex3* fluidForcing, //g
					 cufftComplex3* gridVels, //on exit this is: 1/\eta·\mathcal{L}^-1·g
					 real viscosity,
					 Grid grid){
	/*I expect a 3D grid of threads, one for each fourier node/grid cell*/
	/*Get my cell*/
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;
	if(cell.x>grid.cellDim.x/2+1) return; //I use R2C and C2R ffts
	if(cell.y>=grid.cellDim.y) return;
	if(cell.z>=grid.cellDim.z) return;
	/*Get my cell index (position in the array) */
	const int icell =grid.getCellIndex(cell);
	const int ncells = grid.getNumberCells();
	//k=0 cannot contribute, v_cm = 0
	if(icell==0){
	  gridVels[0] = {0,0 ,0,0 ,0,0};
	  return;
	}
	//Staggered grid requires to define an effective k -> keff_\alpha = 2/d\alpha*sin(k_\alpha*d\alpha/2)
	// that takes into account the cell face shifted velocities in the discrete operators.
	//See eq. 57 in [2].
	real3 keff;
	//sin and cos for a phase of d\alpha/2, a phase of -d\alpha/2 corresponds to (cosk, -sink)
	real3 sink, cosk;
	{
	  const real3 k = cellToWaveNumber(cell, grid.cellDim, grid.box.boxSize);
	  sincos(k.x*grid.cellSize.x*real(0.5), &sink.x, &cosk.x);
	  sincos(k.y*grid.cellSize.y*real(0.5), &sink.y, &cosk.y);
	  sincos(k.z*grid.cellSize.z*real(0.5), &sink.z, &cosk.z);
	  keff = real(2.0)*grid.invCellSize*sink;
	}
	//Shift fluid forcing to cell centers
	cufftComplex3 vk = shiftVelocity(fluidForcing[icell], cosk, real(-1.0)*sink);
	{
	  const real invL = real(-1.0)/dot(keff, keff); //\hat{L}^-1 = 1/(ik)^2 = -1/|keff|^2
	  //Project into divergence free space. i.e Apply \mathcal{L}^-1 = P·L^-1 operator. Transforming fluid forcing into fluid velocity
	  const real prefactor = real(-1.0)*invL/viscosity; //Applies -L^-1·\eta^-1
	  vk = projectFourier(keff, prefactor*vk); //Applies P
	}
	//Store new velocity shifted back to cell faces, normalize FFT
	gridVels[icell] = (real(1.0)/real(ncells))*shiftVelocity(vk, cosk, sink);
      }

      enum class Step{PREDICTOR, CORRECTOR, EULER};
      //Computes
      //  q^{n+1/2} = q^n + dt/2 J^n v in predictor mode
      //  q^{n+1} = q^n + dt J^{n+1/2} v in corrector mode
      //  q^{n+1} = q^n + dt J^n v in euler mode
      template<Step mode = Step::PREDICTOR, class Kernel = IBM_kernels::Peskin::threePoint>
      __global__ void midPointStep(real4* pos,    //q^n in predictor and euler, q^{n+1/2} in corrector
				   real4* posOld, //empty in predictor and euler, q^n in corrector
				   const real3* gridVels,
				   Grid grid,
				   Kernel kernel,
				   real dt,
				   int numberParticles){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= numberParticles) return;
	//Pos prediction
	real3 pnew = make_real3(0);
	real3 posCurrent = make_real3(pos[id]);
	//Store q^n
	if(mode==Step::PREDICTOR) posOld[id] = pos[id];
	//Staggered grid cells for interpolating each velocity coordinate
	const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);
	constexpr int supportCells = Kernel::support;
	const real3 pix = make_real3(posCurrent.x - real(0.5)*grid.cellSize.x, posCurrent.y, posCurrent.z);
	const int3 cellix = grid.getCell(pix);
	const int3 Px = computeSupportShift(pix, cellix, grid, supportCells);
	const real3 piy =  make_real3(posCurrent.x, posCurrent.y - real(0.5)*grid.cellSize.y, posCurrent.z);
	const int3 celliy = grid.getCell(piy);
	const int3 Py = computeSupportShift(piy, celliy, grid, supportCells);
	const real3 piz = make_real3(posCurrent.x, posCurrent.y, posCurrent.z - real(0.5)*grid.cellSize.z);
	const int3 celliz = grid.getCell(piz);
	const int3 Pz = computeSupportShift(piz, celliz, grid, supportCells);
	real prefactor = dt;
	//Half step in predictor mode
	if(mode==Step::PREDICTOR) prefactor *= real(0.5);
	constexpr int numberNeighbourCells = supportCells*supportCells*supportCells;
	//J = dV·S
	const real dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
	//Sum contribution of neighbouring cells
	for(int i = 0; i<numberNeighbourCells; i++){
	  //Apply J to the velocity of the current cell.
	  {
	    //Neighbour cell in the x direction
	    int3 celljx = make_int3(cellix.x + i%supportCells,
				    cellix.y + (i/supportCells)%supportCells,
				    cellix.z + i/(supportCells*supportCells)) - Px;
	    celljx = grid.pbc_cell(celljx);
	    //Cel lindex of neighbour
	    const int jcellx = grid.getCellIndex(celljx);
	    //Distance from particle i to center of cell j
	    //p += J·v = dV·\delta(p_x_i-cell_x_j)·v_x_j
	    const real v_jx = gridVels[jcellx].x;
	    const real3 r = grid.distanceToCellCenter(posCurrent - make_real3(real(0.5)*grid.cellSize.x, 0, 0),
						      celljx);
	    pnew.x += kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*v_jx*dV;
	  }
	  {
	    int3 celljy = make_int3(celliy.x + i%supportCells,
				    celliy.y + (i/supportCells)%supportCells,
				    celliy.z + i/(supportCells*supportCells)) - Py;
	    celljy = grid.pbc_cell(celljy);
	    const int jcelly = grid.getCellIndex(celljy);
	    const real v_jy = gridVels[jcelly].y;
	    const real3 r = grid.distanceToCellCenter(posCurrent - make_real3(0, real(0.5)*grid.cellSize.y, 0),
						      celljy);
	    pnew.y += kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*v_jy*dV;
	  }
	  {
	    int3 celljz = make_int3(celliz.x + i%supportCells,
				    celliz.y + (i/supportCells)%supportCells,
				    celliz.z + i/(supportCells*supportCells)) - Pz;
	    celljz = grid.pbc_cell(celljz);
	    const int jcellz = grid.getCellIndex(celljz);
	    const real v_jz = gridVels[jcellz].z;
	    const real3 r = grid.distanceToCellCenter(posCurrent - make_real3(0, 0, real(0.5)*grid.cellSize.z),
							      celljz);
	    pnew.z += kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*v_jz*dV;
	  }
	}
	//Update position
	if(mode==Step::PREDICTOR){
	  pnew = posCurrent + prefactor*pnew; // p^{n+1/2} = p^n + dt/2 J^n v
	  pos[id] = make_real4(pnew, pos[id].w);
	}
	else if(mode==Step::CORRECTOR || mode==Step::EULER){
	  // p^{n+1} = p^n + dt J^{n+1/2} v in corrector
	  // p^{n+1} = p^n + dt J^n v in euler
	  pnew = make_real3(posOld[id]) + prefactor*pnew;
	  //Write to global memory
	  pos[id] = make_real4(pnew, posOld[id].w);
	}
      }
    }
    //v += S·F. Computes the force acting on the particles and applies the spreading operator to it.
    void FIB::spreadParticleForces(){
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128; /*threads per block*/
      int nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int nblocks = numberParticles/nthreads +  ((numberParticles%nthreads!=0)?1:0);
      {
	auto force = pd->getForce(access::location::gpu, access::mode::write);
	auto force_gr = pg->getPropertyIterator(force);
	thrust::fill(thrust::cuda::par, force_gr, force_gr + numberParticles, real4());
      }
      /*Compute new force*/
      for(auto forceComp: interactors) forceComp->sum({.force = true, .energy= false, .virial = false}, 0);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      auto indexIter = pg->getIndexIterator(access::location::gpu);
      FIB_ns::spreadParticleForces<<<nblocks, nthreads>>>(pos.raw(),
       							  force.raw(),
       							  d_gridVels,
       							  indexIter,
       							  grid,
       							  numberParticles,
       							  *kernel);
    }
    //v += prefactor·\tilde{D}·W
    //Prefactor will be proportional to sqrt(4*viscosity*temperature/(dt*dV)) depending on the integration method
    //This function will use the current contents of the random array and will not change it in any way
    void FIB::randomAdvection(real noisePrefactor){
      if(temperature==real(0.0)) return;
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      //Compute and sum the stochastic advection
      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;
      NblocksCells.x = grid.cellDim.x/NthreadsCells.x + ((grid.cellDim.x%NthreadsCells.x)?1:0);
      NblocksCells.y = grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
      NblocksCells.z = grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
      FIB_ns::addRandomAdvection<<<NblocksCells, NthreadsCells>>>(d_gridVels,
								  grid,
								  noisePrefactor,
								  (real*) thrust::raw_pointer_cast(random.data()));
    }

    // v +=  kbT/\delta [ S(q^n + \delta/2\hat{W}^n) - S(q^n - \delta/2\hat{W}^n) ] ·\hat{W}^n
    //See eq. 32 and 33 in [1]
    void FIB::thermalDrift(){
      if(temperature==real(0.0)) return;
      int numberParticles = pg->getNumberParticles();
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      real driftPrefactor = temperature/deltaRFD;
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      FIB_ns::addThermalDrift<<<Nblocks, Nthreads>>>(pos.raw(),
						     d_gridVels,
						     grid,
						     driftPrefactor,
						     deltaRFD,
						     numberParticles,
						     *kernel,
						     (uint)seed, (uint)step);
    }

    //Takes \vec{g} = S·F + noise + thermalDrift and transforms it to v = 1/\eta\mathcal{L}^-1\vec{g} in Fourier space
    //See eq. 28 and beyond in [1].
    void FIB::applyStokesSolutionOperator(){
      cufftReal* d_gridVels = (cufftReal*)thrust::raw_pointer_cast(gridVels.data());
      cufftComplex* d_gridVelsFourier = (cufftComplex*)thrust::raw_pointer_cast(gridVelsFourier.data());
      //Go to fourier space
      sys->log<System::DEBUG3>("[BDHI::FIB] Taking grid to wave space");
      CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, d_gridVels, d_gridVelsFourier));
      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;
      NblocksCells.x = grid.cellDim.x/NthreadsCells.x + ((grid.cellDim.x%NthreadsCells.x)?1:0);
      NblocksCells.y = grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
      NblocksCells.z = grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
      //Solve Stokes
      sys->log<System::DEBUG3>("[BDHI::FIB] Applying Fourier stokes operator.");
      FIB_ns::solveStokesFourier<<<NblocksCells, NthreadsCells>>>((cufftComplex3*)d_gridVelsFourier,
								  (cufftComplex3*)d_gridVelsFourier,
								  viscosity,
								  grid);
      sys->log<System::DEBUG3>("[BDHI::FIB] Going back to real space");
      //Go back to real space
      CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse,d_gridVelsFourier, d_gridVels));
    }

    //Updates positions to half step according to the current fluid velocities
    //q^{n+1/2} = q^n + dt/2 J^n v
    void FIB::predictorStep(){
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      real4 * d_posOld = thrust::raw_pointer_cast(posOld.data());
      FIB_ns::midPointStep<FIB_ns::Step::PREDICTOR><<<Nblocks, Nthreads>>>(pos.raw(),
									   d_posOld,
									   d_gridVels,
									   grid,
									   *kernel,
									   dt,
									   numberParticles);
    }
    //Updates positions to the next step according to the current fluid velocities
    //q^{n+1} = q^n + dt J^{n+1/2} v
    void FIB::correctorStep(){
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      real4 * d_posOld = thrust::raw_pointer_cast(posOld.data());
      FIB_ns::midPointStep<FIB_ns::Step::CORRECTOR><<<Nblocks, Nthreads>>>(pos.raw(),
									   d_posOld,
									   d_gridVels,
									   grid,
									   *kernel,
									   dt,
									   numberParticles);
    }
    //Updates positions to the next step according to the current fluid velocities
    //q^{n+1} = q^n + dt J^n v
    void FIB::eulerStep(){
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      real4 * d_posOld = thrust::raw_pointer_cast(posOld.data());
      FIB_ns::midPointStep<FIB_ns::Step::EULER><<<Nblocks, Nthreads>>>(pos.raw(),
								       d_posOld,
								       d_gridVels,
								       grid,
								       *kernel,
								       dt,
								       numberParticles);
    }

    //forwards the simulation to the next time step with the simple midpoint scheme in [1]
    void FIB::forwardMidpoint(){
      sys->log<System::DEBUG2>("[BDHI::FIB] Reset fluid velocity");
      thrust::fill(gridVels.begin(), gridVels.end(), real3());
      sys->log<System::DEBUG2>("[BDHI::FIB] Random advection");
      if(temperature!=real(0.0)){
       	sys->log<System::DEBUG2>("[BDHI::FIB] Generate random numbers");
       	CurandSafeCall(curandgeneratenormal(curng,
       					    thrust::raw_pointer_cast(random.data()),
       					    random.size(),
       					    0.0, 1.0));
	//sqrt(2·vis·kT/(dt·dV))·\hat{D}\bf{W}^{n,1}
       	double dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
       	real noisePrefactor = sqrt(2*viscosity*temperature/(dt*dV));
       	randomAdvection(noisePrefactor);
      }
      sys->log<System::DEBUG2>("[BDHI::FIB] Thermal drift");
      //kT/\delta [ S(q^{n+1/2} + \delta/2\hat{\bf{W}}^{n+1/2}) - S(q^{n+1/2} - \delta/2\hat{\bf{W}}^{n+1/2})]\hat{\bf{W}}^{n+1/2}
      //Spread thermal drift with RFD
      thermalDrift();
      sys->log<System::DEBUG2>("[BDHI::FIB] Spread particle forces");
      //S^n·F^n
      spreadParticleForces();
      sys->log<System::DEBUG2>("[BDHI::FIB] Solve fluid");
      //v = vis^-1\mathcal{\bf{L}}^-1·g
      applyStokesSolutionOperator();
      sys->log<System::DEBUG2>("[BDHI::FIB] Predictor step");
      //q^{n+1/2} = q^n + dt/2·J^n·v
      predictorStep();
      sys->log<System::DEBUG2>("[BDHI::FIB] Corrector");
      // //q^n+1 = q^n + dt·J^{n+1/2}·v
      correctorStep();
      for(auto updatable: updatables) updatable->updateSimulationTime((step+1)*dt);
    }

    //forwards the simulation to the next time step with the improved midpoint scheme in [1]
    void FIB::forwardImprovedMidpoint(){
      sys->log<System::DEBUG2>("[BDHI::FIB] Reset fluid velocity");
      thrust::fill(gridVels.begin(), gridVels.end(), real3());
      sys->log<System::DEBUG2>("[BDHI::FIB] Random advection");
      if(temperature!=real(0.0)){
	sys->log<System::DEBUG2>("[BDHI::FIB] Generate random numbers");
	CurandSafeCall(curandgeneratenormal(curng,
					    thrust::raw_pointer_cast(random.data()),
					    random.size(),
					    0.0, 1.0));
      //sqrt(4·vis·kT/(dt·dV))·\hat{D}\bf{W}^{n,1}
	double dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
	real noisePrefactor = sqrt(4*viscosity*temperature/(dt*dV));
	randomAdvection(noisePrefactor);
      }
      sys->log<System::DEBUG2>("[BDHI::FIB] Spread particle forces");
      //S^n·F^n
      spreadParticleForces();
      sys->log<System::DEBUG2>("[BDHI::FIB] Solve fluid");
      //v = vis^-1\mathcal{\bf{L}}^-1·g
      applyStokesSolutionOperator();
      sys->log<System::DEBUG2>("[BDHI::FIB] Predictor step");
      //q^{n+1/2} = q^n + dt/2·J^n·v
      predictorStep();
      for(auto updatable: updatables) updatable->updateSimulationTime((step+0.5)*dt);
      //Clean velocity for next half step
      thrust::fill(gridVels.begin(), gridVels.end(), real3());
      //sqrt(vis·kT/(dt·dV))·\hat{D}(\bf{W}^{n,1}+\bf{W}^{n,2})
      {
	double dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
	real noisePrefactor = sqrt(viscosity*temperature/(dt*dV));
	if(temperature!=real(0.0))
	  CurandSafeCall(curandgeneratenormal(curng,
					      thrust::raw_pointer_cast(random.data()),
					      random.size(),
					      0.0, 1.0));
	randomAdvection(noisePrefactor);
      }
      sys->log<System::DEBUG2>("[BDHI::FIB] Thermal drift");
      //kT/\delta [ S(q^{n+1/2} + \delta/2\hat{\bf{W}}^{n+1/2}) - S(q^{n+1/2} - \delta/2\hat{\bf{W}}^{n+1/2})]\hat{\bf{W}}^{n+1/2}
      //Spread thermal drift with RFD
      thermalDrift();
      sys->log<System::DEBUG2>("[BDHI::FIB] Spread particle forces");
      //S^{n+1/2}·F^{n+1/2}
      spreadParticleForces();
      sys->log<System::DEBUG2>("[BDHI::FIB] Solve fluid");
      //v = vis^-1\mathcal{\bf{L}}^-1·g
      applyStokesSolutionOperator();
      sys->log<System::DEBUG2>("[BDHI::FIB] Corrector");
      //q^n+1 = q^n + dt·J^{n+1/2}·v
      correctorStep();
      for(auto updatable: updatables) updatable->updateSimulationTime((step+1)*dt);
    }

    //Takes the simulation to the nxt time step
    void FIB::forwardTime(){
      CudaCheckError();
      step++;
      if(step==1){
	for(auto updatable: updatables){
	  updatable->updateSimulationTime(0);
	  updatable->updateViscosity(viscosity);
	  updatable->updateTimeStep(dt);
	  updatable->updateTemperature(temperature);
	  updatable->updateBox(box);
	}
      }
      sys->log<System::DEBUG1>("[BDHI::FIB] Performing step");
      switch(scheme){
      case Scheme::MIDPOINT:
	this->forwardMidpoint();break;
      case Scheme::IMPROVED_MIDPOINT:
	this->forwardMidpoint();break;
      }
    }

    real FIB::sumEnergy(){
      //Sum 1.5*kT to each particle
      auto energy = pd->getEnergy(access::gpu, access::readwrite);
      auto energy_gr = pg->getPropertyIterator(energy);
      auto energy_per_particle = thrust::make_constant_iterator<real>(1.5*temperature);
      thrust::transform(thrust::cuda::par,
			energy_gr, energy_gr + pg->getNumberParticles(),
			energy_per_particle,
			energy_gr,
			thrust::plus<real>());
      return 0;
    }
  }
}

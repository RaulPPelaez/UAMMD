#include"FIB.cuh"
#include"utils/cufftDebug.h"
#include"utils/debugTools.h"

namespace uammd{
  namespace BDHI{

    namespace FIB_ns{
      //Looks for the closest (equal or greater) number of nodes of the form 2^a*3^b*5^c
      int3 nextFFTWiseSize3D(int3 size){
	int* cdim = &size.x;
	/*Store up to 2^5·3^5·5^5 in tmp*/    
	int n= 5;
	std::vector<int> tmp(n*n*n, 0);
	int max_dim = std::max({size.x, size.y, size.z});
    
	do{
	  tmp.resize(n*n*n, 0);
	  fori(0,n)forj(0,n)for(int k=0; k<n;k++)
	    tmp[i+n*j+n*n*k] = pow(2,i)*pow(3,j)*pow(5,k);
	  n++;
	  /*Sort this array in ascending order*/
	  std::sort(tmp.begin(), tmp.end());      
	}while(tmp.back()<max_dim); /*if n=5 is not enough, include more*/
	
	/*Now look for the nearest value in tmp that is greater than each cell dimension*/
	forj(0,3){
	  int i = 0;
	  while(tmp[i]<cdim[j]) i++;
	  cdim[j] = tmp[i];
	}
	return size;
      }

      namespace PeskinKernel{
	
	struct threePoint{
	  real invh;
	  constexpr int support = 3;
	  threePoint(real h):invh(1.0/h){}
	  inline __device__ real phi(real r){
	    if(r<real(0.5)){
	      constexpr real onediv6 = 1/6.0;
	      const real omr = real(1.0) - r;
	      return onediv6*(real(5.0)-real(3.0)*r - sqrt(real(1.0) + real(-3.0)*omr*omr));
	    }
	    else if(r<real(1.5)){
	      constexpr real onediv3 = real(1/3.0);
	      return onediv3*(real(1.0) + sqrt(real(1.0)+real(-3.0)*r*r));	      
	    }
	    else return 0;	    
	  }	  
	  inline __device__ real delta(real3 rvec){	    
	    return invh*invh*invh*phi(rvec.x*invh)*phi(rvec.y*invh)*phi(rvec.z*invh);	    
	  }
	  
	};

      }
    }
    
    FIB::FIB(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys,		       
	     Parameters par):
      Integrator(pd, pg, sys, "BDHI::FIB"),
      dt(par.dt),
      temperature(par.temperature),
      box(par.box),
      deltaRFD(1e-4){
      
      CudaCheckError();
      sys->log<System::MESSAGE>("[BDHI::FIB] Initialized");

      seed = sys->rng().next();
      
      int numberParticles = pg->getNumberParticles();


      double hgrid = 0.91*par.hydrodynamicRadius;
      int3 cellDim = make_int3(box.boxSize/hgrid);

      if(par.cells.x>0) cellDim = par.cells;

      sys->log<System::MESSAGE>("[BDHI::FIB] Target hydrodynamic radius: %f", par.hydrodynamicRadius);
      double rh = 0.91*boxSize.x/cellDim.x;
      sys->log<System::MESSAGE>("[BDHI::FIB] Closest possible hydrodynamic radius: %f", rh);
      sys->log<System::MESSAGE>("[BDHI::FIB] Self mobility: %f", 1.0/(6*M_PI*par.viscosity*rh)*(1-2.837297*rh/box.boxSize.x));
      
      if(cellDim.x<3)cellDim.x = 3;
      if(cellDim.y<3)cellDim.y = 3;
      if(cellDim.z==2)cellDim.z = 3; 

      /*FFT likes a number of cells as cellDim.i = 2^n·3^l·5^m */      
      cellDim = FIB_ns::nextFFTWiseSize3D(cellDim);
      
      /*Store grid parameters in a Mesh object*/
      this->grid = Grid(box, cellDim);

      /*Print information*/
      sys->log<System::MESSAGE>("[BDHI::FIB] Box Size: %f %f %f", box.boxSize.x, box.boxSize.y, box.boxSize.z);
        
      /*The quantity spreaded to the grid in real or wave space*/
      /*The layout of this array is
	fx000, fy000, fz000, fx001, fy001, fz001..., fxnnn, fynnn, fznnn. n=ncells-1
	When used in real space each f is a real number, whereas in wave space each f will be a complex number.
      */
      int ncells = grid.cellDim.x*grid.cellDim.y*grid.cellDim.z;
      gridVelsFourier.resize(3*ncells, cufftComplex());


      int numberParticles = pg->getNumberParticles();
      
      posPrediction.resize(numberParticles);



      //Init rng
      CurandSafeCall(curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT));
      //curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_MT19937);
    
      CurandSafeCall(curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next()));
      {
	thrust::device_vector<real> noise(30000);
	auto noise_ptr = thrust::raw_pointer_cast(noise.data());
	//Warm cuRNG
	CurandSafeCall(curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0));
	CurandSafeCall(curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0));
      }
      random.resize(6*ncells);

      CufftSafeCall(cufftCreate(&cufft_plan_forward));
      CufftSafeCall(cufftCreate(&cufft_plan_inverse));
      
      /*I will be handling workspace memory*/
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));

      //Required storage for the plans
      size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
      /*Set up cuFFT*/
      int3 cdtmp = {grid.cellDim.x, grid.cellDim.y, grid.cellDim.z};
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
	sys->log<System::CRITICAL>("[BDHI::FIB] Not enough memory in device to allocate cuFFT free %s, needed: %s!!, try lowering the splitting parameter!",
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(cufftWorkSize).c_str());
      }

      cufftWorkArea.resize(cufftWorkSize+1);
      void * d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      
      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, d_cufftWorkArea));
  
      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
    }
      				     
    ~FIB::FIB(){
      CudaCheckError();
      CudaSafeCall(cudaDeviceSynchronize());
      CufftSafeCall(cufftDestroy(cufft_plan_inverse));
      CufftSafeCall(cufftDestroy(cufft_plan_forward));
      CudaCheckError();
    }


    namespace FIB_ns{

      enum class Direction{ XX = 0, YY, ZZ, XY, XZ, YZ};
      
      //Adds the stochastic term to the current velocity: v_i += noisePrefactor*\hat{D}·W^n
      //W is a symmetric tensor with 3 white gaussian numbers per direction
      //(staggered grid, W is defined at the cell centers and edges and \hat{D}·W in the faces)
      //W^{\alpha\alpha} are in the centers with variance 2
      //W^{\alpha\beta} are in the edges with variance 1
      __global__ void addRandomAdvection(real3* gridVels,
					 Grid grid,
					 real noisePrefactor,
					 real* random, //6 random numbers per cell, sorted by direction [first number for all cells, ..., second number for all cells,...]
					 uint seed, uint step){
	/*I expect a 3D grid of threads, one for each fourier node/grid cell*/
	/*Get my cell*/
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;

	/*Get my cell index (position in the array) */
	int icell =grid.getCellIndex(cell);
	if(cell.x>=grid.cellDim.x) return;
	if(cell.y>=grid.cellDim.y) return;
	if(cell.z>=grid.cellDim.z) return;

	//I will draw 6 random numbers for each cell
	
	const int ncells = grid.getNumberCells();
	real3 DW = make_real3(0);
	//(\nabla·W)_\alpha^i = \nabla^i·(W_{\alpha x}, W_{\alpha y}, W_{\alpha z}) ->
	// -> \partial_\alpha^i W_{\alpha\beta} = 1/d\alpha (W^{\alpha\beta}_{i+\alpha/2} - W^{\alpha\beta}_{i-\alpha/2})
	constexpr real sqrt2 = sqrt(2);
	//Diagonal terms
	{ //m -> \alpha - 1/2, p -> \alpha + 1/2
	  const int n_mx = icell; //n_[m/p]\alpha -> cell index of neighbour in a certain direction	  
	  real wxx_mx = sqrt2*random[n_mx+ncells*Direction::XX];
	  
	  const int n_px = grid.getCellIndex(grid.pbc_cells({cell.x + 1, cell.y, cell.z}));	  
	  real wxx_px = sqrt2*random[n_px+ncells*Direction::XX];
	  
	  DW.x += grid.invCellSize.x*(wxx_px - wxx_mx);
	}
	{
	  const int n_my = icell;  
	  real wyy_my = sqrt2*random[n_my+ncells*Direction::YY];
	  
	  const int n_py = grid.getCellIndex(grid.pbc_cells({cell.x, cell.y + 1, cell.z}));
	  real wyy_py = sqrt2*random[n_py+ncells*Direction::YY];
	  
	  DW.y += grid.invCellSize.y*(wyy_py - wyy_my);
	}
	{
	  const int n_mz = icell;  
	  real wzz_mz = sqrt2*random[n_mz+ncells*Direction::ZZ];
	  
	  const int n_pz = grid.getCellIndex(grid.pbc_cells({cell.x, cell.y, cell.z + 1}));
	  real wzz_pz = sqrt2*random[n_pz+ncells*Direction::ZZ];
	  
	  DW.z += grid.invCellSize.z*(wzz_pz - wzz_mz);
	}
	//Cross terms
	{ //W is simmetric so wxy_mx = wyx_my, etc
	  const int n_my = icell;  
	  real wxy_m = saru.gf(0, 1.0f);
	  real wxy_p = saru.gf(0, 1.0f);
	  real dwxy_d = (wxy_p - wxy_m);
	  DW.x += grid.invCellSize.y*dwxy_d; //dy
	  DW.y += grid.invCellSize.x*dwxy_d; //dx
	}
	{
	  real wxz_m = saru.gf(0, 1.0f);
	  real wxz_p = saru.gf(0, 1.0f);
	  real dwxz_d = (wxz_p - wxy_m);
	  DW.x += grid.invCellSize.z*dwxz_d;
	  DW.z += grid.invCellSize.x*dwxz_d;
	}
	
	{
	  real wyz_m = saru.gf(0, 1.0f);
	  real wyz_p = saru.gf(0, 1.0f);
	  real dwyz_d = (wyz_p - wyy_m);
	  DW.y += grid.invCellSize.z*dwyz_d;
	  DW.z += grid.invCellSize.y*dwyz_d;
	}

	
	
      }

      


    }

    void FIB::spreadParticleForces(){


    }
    void FIB::randomAdvection(){
      auto d_gridVels = thrust::raw_pointer_cast(gridVelsFourier.data());
      
      //Compute and sum the stochastic advection
      double dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;	
      real noisePreFactor = sqrt(2.0*viscosity*temperature/(dt*dV));

      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;		  
      NblocksCells.x = grid.cellDim.x/NthreadsCells.x + ((grid.cellDim.x%NthreadsCells.x)?1:0);
      NblocksCells.y = grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
      NblocksCells.z = grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
	
      addRandomAdvection<<<NblocksCells, NthreadsCells>>>((real3*)d_gridVels,
							  grid,
							  noisePrefactor,
							  (uint)seed, (uint)step);   
    }

    void FIB::thermalDrift(){
      auto d_gridVels = thrust::raw_pointer_cast(gridVelsFourier.data());
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);

      real driftPrefactor = temperature/deltaRFD;

      real4 * pos = pd->getPos(access::location::gpu, access::mode::read);

      addThermalDrift<<<Nblocks, Nthreads>>>(pos,
					     (real3*)d_gridVels,
					     grid,
					     driftPrefactor,
					     numberParticles,
					     (uint)seed, (uint)step);	
    }

    void FIB::applyStokesSolutionOperator(){      
      auto d_gridVels = thrust::raw_pointer_cast(gridVelsFourier.data());
      
      sys->log<System::DEBUG2>("[BDHI::FIB] Taking grid to wave space");      
      auto cufftStatus =
	cufftExecReal2Complex(cufft_plan_forward,
		     (cufftReal*)d_gridVels,
		     (cufftComplex*)d_gridVelsFourier);
      if(cufftStatus != CUFFT_SUCCESS){
	sys->log<System::CRITICAL>("[BDHI::FIB] Error in forward CUFFT");
      }

      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;		  
      NblocksCells.x = grid.cellDim.x/NthreadsCells.x + ((grid.cellDim.x%NthreadsCells.x)?1:0);
      NblocksCells.y = grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
      NblocksCells.z = grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);

      solveStokesFourier<<<NblocksCells, NthreadsCells>>>((cufftComplex3*)d_gridVels,
							  (cufftComplex3*)d_gridVels,
							  grid);
							  
      
      sys->log<System::DEBUG2>("[BDHI::FIB] Going back to real space");
      
      cufftStatus =
	cufftExecComplex2Real(cufft_plan_inverse,
		     (cufftComplex*)d_gridVelsFourier,
		     (cufftReal*)d_gridVels);
      if(cufftStatus != CUFFT_SUCCESS){
	sys->log<System::CRITICAL>("[BDHI::FIB] Error in inverse CUFFT");
      }
  }

    
    void FIB::predictorStep(){
      int numberParticles = pg->getNumberParticles();
      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;		  
      NblocksCells.x = grid.cellDim.x/NthreadsCells.x + ((grid.cellDim.x%NthreadsCells.x)?1:0);
      NblocksCells.y = grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
      NblocksCells.z = grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);

      real4 * pos = pd->getPos(access::location::gpu, access::mode::read);
      real3 * d_posPrediction = thrust::raw_pointer_cast(posPrediction.data());
      predictorStep<<<Nblocks, Nthreads>>>(pos,
					   d_posPrediction,
					   (real3*) d_gridVels,
					   grid,
					   driftPrefactor,
					   numberParticles,
					   (uint)seed, (uint)step);	     
    }
    void FIB::correctorStep(){
      int numberParticles = pg->getNumberParticles();
      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;		  
      NblocksCells.x = grid.cellDim.x/NthreadsCells.x + ((grid.cellDim.x%NthreadsCells.x)?1:0);
      NblocksCells.y = grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
      NblocksCells.z = grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);

      real4 * pos = pd->getPos(access::location::gpu, access::mode::read);
      real3 * d_posPrediction = thrust::raw_pointer_cast(posPrediction.data());
      correctorStep<<<Nblocks, Nthreads>>>(pos,
					   d_posPrediction,
					   (real3*) d_gridVels,
					   grid,
					   driftPrefactor,
					   numberParticles,
					   (uint)seed, (uint)step);
    }
    
    void FIB::forwardTime(){
      CudaCheckError();
      step++;
      CurandSafeCall(curandGenerateNormal(curng, thrust::raw_pointer_cast(random.data()), random.size(), 0.0, 1.0));
      //g = S^n·F^n + sqrt(2·vis·kT/(dt·dV))·\hat{D}\bf{W}^{n,1} + kT/\delta [ S(q^n + \delta/2\hat{\bf{W}}^n) - S(q^n - \delta/2\hat{\bf{W}}^n)]\hat{\bf{W}}^n
      //S^n·F^n
      spreadParticleForces();
      //sqrt(2·vis·kT/(dt·dV))·\hat{D}\bf{W}^{n,1}
      randomAvdection();
      //kT/\delta [ S(q^n + \delta/2\hat{\bf{W}}^n) - S(q^n - \delta/2\hat{\bf{W}}^n)]\hat{\bf{W}}^n
      //Spread thermal drift with RFD
      thermalDrift();
      
      //v = vis^-1\mathcal{\bf{L}}^-1·g
      applyStokesSolutionOperator();

      //q^{n+1/2} = q^n + dt/2·J^n·v
      predictorStep();
      //q^n+1 = q^n + dt·J^{n+1/2}·v
      correctorStep();
    }
    real FIB::sumEnergy(){ return 0;}
         
  
}
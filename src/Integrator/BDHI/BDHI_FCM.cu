/*Raul P. Pelaez 2018. Force Coupling Method BDHI Module.
  See BDHI_FCM.cuh for information.
*/
#include"BDHI_FCM.cuh"
#include"utils/GPUUtils.cuh"
#include"third_party/saruprng.cuh"
#include<vector>
#include<algorithm>
#include<fstream>
#include"utils/debugTools.cuh"
#include"utils/cufftDebug.h"
#include"utils/cxx_utils.h"

namespace uammd{
  namespace BDHI{

    FCM::FCM(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys,
	     Parameters par):
      pd(pd), pg(pg), sys(sys),
      dt(par.dt),
      temperature(par.temperature),
      viscosity(par.viscosity),
      box(par.box), grid(box, int3()){

      seed = sys->rng().next();
      sys->log<System::MESSAGE>("[BDHI::FCM] Initialized");

      if(box.boxSize.x == real(0.0) && box.boxSize.y == real(0.0) && box.boxSize.z == real(0.0)){
	sys->log<System::CRITICAL>("[BDHI::FCM] Box of size zero detected, cannot work without a box! (make sure a box parameter was passed)");
      }


      sys->log<System::WARNING>("[BDHI::FCM] Current implementation ignores tolerance parameter, it is over protective and always sets precision to a very low tolerance.");
      
      int numberParticles = pg->getNumberParticles();

      double rh = par.hydrodynamicRadius;
      
      double L = par.box.boxSize.x;
      long double M0 = 1.0l/(6.0l*M_PIl*viscosity*rh)*(1.0l-2.837297l*rh/L+(4.0l/3.0l)*M_PIl*pow(rh/L,3.0)-27.4l*pow(rh/L,6.0l));
      sys->log<System::MESSAGE>("[BDHI::FCM] Self mobility: %g", (double)M0);
      
      if(box.boxSize.x != box.boxSize.y || box.boxSize.y != box.boxSize.z || box.boxSize.x != box.boxSize.z){
	sys->log<System::WARNING>("[BDHI::FCM] Self mobility will be different for non cubic boxes!");
      }
	
      
      real sigma = par.hydrodynamicRadius/sqrt(M_PI); //eq. 8 in [1], \sigma_\Delta
      int3 cellDim;
      if(par.cells.x<=0){
	double minFactor = 1.5; //According to [1] \sigma_\Delta/H = 1.86 gives enough accuracy
	real h = sigma/minFactor;
	//real h = par.hydrodynamicRadius;
	cellDim = nextFFTWiseSize3D(make_int3(box.boxSize/h));
      }
      else{      
        cellDim = par.cells;
      }
      grid = Grid(box, cellDim);
      	
      Kernel kernel(grid.cellSize);
      
      sys->log<System::MESSAGE>("[BDHI::FCM] Box Size: %g %g %g", grid.box.boxSize.x, grid.box.boxSize.y, grid.box.boxSize.z);
      sys->log<System::MESSAGE>("[BDHI::FCM] Grid dimensions: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
      sys->log<System::MESSAGE>("[BDHI::FCM] Interpolation kernel support: %g rh, %d cells", kernel.support*grid.cellSize.x, kernel.support);      
      sys->log<System::MESSAGE>("[BDHI::FCM] σ_Δ: %g", sigma);
      sys->log<System::MESSAGE>("[BDHI::FCM] h: %g %g %g", grid.cellSize.x, grid.cellSize.y, grid.cellSize.z);
      sys->log<System::MESSAGE>("[BDHI::FCM] σ_Δ/h: %g %g %g", sigma*grid.invCellSize.x, sigma*grid.invCellSize.y, sigma*grid.invCellSize.z);
      sys->log<System::MESSAGE>("[BDHI::FCM] Cell volume: %e", grid.cellSize.x*grid.cellSize.y*grid.cellSize.z);
      
      CudaSafeCall(cudaStreamCreate(&stream));
      CudaSafeCall(cudaStreamCreate(&stream2));
  
      /*The quantity spreaded to the grid in real or wave space*/
      /*The layout of this array is
	fx000, fy000, fz000, fx001, fy001, fz001..., fxnnn, fynnn, fznnn. n=ncells-1
	When used in real space each f is a real number, whereas in wave space each f will be a complex number.
	See cufftC2R of R2C in place in Mdot_far
      */
      /*Can be Force when spreading particles to the grid and
	velocities when interpolating from the grid to the particles*/
      int ncells = grid.cellDim.x*grid.cellDim.y*grid.cellDim.z;
      gridVelsFourier.resize(3*ncells, cufftComplex());
      
      initCuFFT();
  
      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
    }

    void FCM::initCuFFT(){
      CufftSafeCall(cufftCreate(&cufft_plan_forward));
      CufftSafeCall(cufftCreate(&cufft_plan_inverse));
      
      /*I will be handling workspace memory*/
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
      CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));

      //Required storage for the plans
      size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
      /*Set up cuFFT*/
      //This sizes have to be reversed according to the cufft docs
      int3 cdtmp = {grid.cellDim.z, grid.cellDim.y, grid.cellDim.x};
      int3 inembed = {grid.cellDim.z, grid.cellDim.y, grid.cellDim.x};
      /*I want to make three 3D FFTs, each one using one of the three interleaved coordinates*/
      CufftSafeCall(cufftMakePlanMany(cufft_plan_forward,
				      3, &cdtmp.x, /*Three dimensional FFT*/
				      &inembed.x,
				      /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
				      3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
				      /*Same format in the output*/
				      &inembed.x,
				      3, 1,
				      /*Perform 3 direct Batched FFTs*/
				      CUFFT_R2C, 3,
				      &cufftWorkSizef));

      sys->log<System::DEBUG>("[BDHI::FCM] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
      /*Same as above, but with C2R for inverse FFT*/
      CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
				      3, &cdtmp.x, /*Three dimensional FFT*/
				      &inembed.x,
				      /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
				      3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
				      &inembed.x,
				      3, 1,
				      /*Perform 3 inverse batched FFTs*/
				      CUFFT_C2R, 3,
				      &cufftWorkSizei));

      /*Allocate cuFFT work area*/
      size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei);
      size_t free_mem, total_mem;
      CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
      
      sys->log<System::DEBUG>("[BDHI::FCM] Necessary work space for cuFFT: %s, available: %s, total: %s",
			      printUtils::prettySize(cufftWorkSize).c_str(),
			      printUtils::prettySize(free_mem).c_str(),
			      printUtils::prettySize(total_mem).c_str());

      if(free_mem<cufftWorkSize){
	sys->log<System::CRITICAL>("[BDHI::FCM] Not enough memory in device to allocate cuFFT free %s, needed: %s!!",
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(cufftWorkSize).c_str());
      }

      cufftWorkArea.resize(cufftWorkSize+1);
      auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      
      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
    }

    FCM::~FCM(){
      CudaCheckError();
      CudaSafeCall(cudaDeviceSynchronize());
      CufftSafeCall(cufftDestroy(cufft_plan_inverse));
      CufftSafeCall(cufftDestroy(cufft_plan_forward));
      CudaSafeCall(cudaStreamDestroy(stream));
      CudaSafeCall(cudaStreamDestroy(stream2));
      CudaCheckError();
    }

    //I dont need to do anything at the begining of a step
    void FCM::setup_step(cudaStream_t st){}

    //Compute M·v = Mw·v
    template<typename vtype>
    void FCM::Mdot(real3 *Mv, vtype *v, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::FCM] Mdot....");
      {
	int numberParticles = pg->getNumberParticles();
	int BLOCKSIZE = 128;
	int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
	int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0); 

	fillWithGPU<<<Nblocks, Nthreads>>>(Mv, make_real3(0.0), numberParticles);
      }
      Mdot_far<vtype>(Mv, v, st);
      
    }
    namespace FCM_ns{
      
      using cufftComplex3 = FCM::cufftComplex3;
      
#ifndef SINGLE_PRECISION
      __device__ double atomicAdd(double* address, double val){
	unsigned long long int* address_as_ull =
	  (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
	  assumed = old;
	  old = atomicCAS(address_as_ull, assumed,
			  __double_as_longlong(val +
					       __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
      }
#endif

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

      /*Apply the projection operator to a wave number with a certain complex factor.
	res = (I-\hat{k}^\hat{k})·factor*/

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
      
      /*Spreads the 3D quantity v (i.e the force) to a regular grid
	For that it uses a Gaussian kernel of the form f(r) = prefactor·exp(-tau·r^2). See eq. 8 in [1]
	i.e. Applies the operator S.
	Launch a block per particle.
      */
      template<class Kernel, typename vtype> /*Can take a real3 or a real4*/
      __global__ void particles2GridD(const real4 * __restrict__ pos, /*Particle positions*/
				      const vtype * __restrict__ v,   /*Per particle quantity to spread*/
				      real3 * __restrict__ gridVels, /*Interpolated values, size ncells*/
				      int N, /*Number of particles*/
				      Grid grid, /*Grid information and methods*/
				      Kernel kernel){
	const int id = blockIdx.x;
	const int tid = threadIdx.x;
	if(id>=N) return;

	/*Get pos and v (i.e force)*/
	__shared__ real3 pi;
	__shared__ real3 vi;
	__shared__ int3 celli;
	if(tid==0){
	  pi = make_real3(pos[id]);
	  vi = make_real3(v[id]);
	  /*Get my cell*/
	  celli = grid.getCell(pi);	  
	}
	/*Conversion between cell number and cell center position*/
	const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);
	const int supportCells = kernel.support;
	const int numberNeighbourCells = supportCells*supportCells*supportCells;
	const int P = supportCells/2;
	__syncthreads();
	for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	  /*Compute neighbouring cell*/
	  int3 cellj = make_int3(celli.x + i%supportCells - P,
				 celli.y + (i/supportCells)%supportCells - P,
				 celli.z + i/(supportCells*supportCells) - P);
	  cellj = grid.pbc_cell(cellj);
	  
	  /*Distance from particle i to center of cell j*/
	  const real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize-cellPosOffset); 
	  /*The weight of particle i on cell j*/
	  const real3 weight = vi*kernel.delta(rij);

	  /*Get index of cell j*/
	  const int jcell = grid.getCellIndex(cellj);
	  
	  /*Atomically sum my contribution to cell j*/
	  atomicAdd(&gridVels[jcell].x, weight.x);
	  atomicAdd(&gridVels[jcell].y, weight.y);
	  atomicAdd(&gridVels[jcell].z, weight.z);	  
	}
      }
      
      /*Scales fourier transformed forces in the regular grid to obtain velocities,
	(Mw·F)_deterministic = σ·St·FFTi·B·FFTf·S·F	
	 Input: gridForces = FFTf·S·F
	 Output:gridVels = B·FFTf·S·F -> B \propto (I-k^k/|k|^2) 
       */
      /*A thread per fourier node*/
      __global__ void forceFourier2Vel(cufftComplex3 * gridForces, /*Input array*/
				       cufftComplex3 * gridVels, /*Output array, can be the same as input*/
				       real vis,				       
				       Grid grid/*Grid information and methods*/
				       ){
	/*Get my cell*/
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;
	/*Only the first half of the innermost dimension is stored, the rest is redundant*/
	if(cell.x>=grid.cellDim.x/2+2) return;
	if(cell.y>=grid.cellDim.y) return;
	if(cell.z>=grid.cellDim.z) return;
	
	const int icell = grid.getCellIndex(cell);
	if(icell == 0){
	  gridVels[0] = {0,0, 0,0, 0,0};
	  return;
	}
	const int ncells = grid.getNumberCells();
	const real3 k = cellToWaveNumber(cell, grid.cellDim, grid.box.boxSize);
	const real invk2 = real(1.0)/dot(k,k);
	/*Get my scaling factor B, Fourier representation of FCM*/
	const real B = invk2/(vis*real(ncells));
	cufftComplex3 factor = gridForces[icell];

	factor.x *= B;
	factor.y *= B;
	factor.z *= B;	
	
	/*Store vel in global memory, note that this is overwritting any previous value in gridVels*/
	gridVels[icell] = projectFourier(k, factor);	  
      }

      /*Computes the long range stochastic velocity term
	Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw = 
	= σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	This kernel gets v_k = gridVelsFourier = B·FFtt·S·F as input and adds 1/√σ·√B(k)·dWw.
	Keeping special care that v_k = v*_{N-k}, which implies that dWw_k = dWw*_{N-k}
	See eq. 30 in [1].
	Launch a thread per cell grid/fourier node
      */
      __global__ void fourierBrownianNoise(/*Values of vels on each cell*/
					   cufftComplex3 *__restrict__ gridVelsFourier, 
					   Grid grid, /*Grid parameters. Size of a cell, number of cells...*/
					   real prefactor,/* sqrt(2·T/(dt·dV))*/
					   real vis,
					   //Parameters to seed the RNG					   
					   ullint seed,
					   ullint step
					   ){
	/*Get my cell*/
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;
	/*This indesx is computed here to use it as a seed for the RNG*/
	int icell = grid.getCellIndex(cell);
	/*cuFFT R2C and C2R only store half of the innermost dimension, the one that varies the fastest
      
	  The input of R2C is real and the output of C2R is real. 
	  The only way for this to be true is if v_k={i,j,k} = v*_k{N-i, N-j, N-k}

	  So the conjugates are redundant and the is no need to compute them nor store them except on two exceptions.
	  In this scheme, the only cases in which v_k and v_{N-k} are stored are:
	     1- When the innermost dimension coordinate is 0.
	     2- When the innermost dimension coordinate is N/2 and N is even.
	*/
	/*Only compute the first half of the innermost dimension*/
	if(2*cell.x >= grid.cellDim.x+1) return;
	if(cell.y >= grid.cellDim.y) return;
	if(cell.z >= grid.cellDim.z) return;

	const int ncells = grid.getNumberCells();
	/*K=0 is not added, no stochastic motion is added to the center of mass*/	
	if((cell.x == 0 and cell.y == 0 and cell.z == 0) or
	   /*These terms will be computed along its conjugates*/
	   /*These are special because the conjugate of k_i=0 is k_i=N_i, 
	     which is not stored and therfore must not be computed*/
	   (cell.x==0 and cell.y == 0 and 2*cell.z >= grid.cellDim.z+1) or
	   (cell.x==0 and 2*cell.y >= grid.cellDim.y+1)) return;
	    
	/*Compute gaussian complex noise dW, 
	  std = prefactor -> ||z||^2 = <x^2>/sqrt(2)+<y^2>/sqrt(2) = prefactor*/
	/*A complex random number for each direction*/
	cufftComplex3 noise;
	{
	  //Uncomment to use uniform numbers instead of gaussian
	  Saru saru(icell, step, seed);
	  const real complex_gaussian_sc = real(0.707106781186547)*prefactor; //1/sqrt(2)
	  //const real sqrt32 = real(1.22474487139159)*prefactor;
	  // = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	  noise.x = make_real2(saru.gf(0, complex_gaussian_sc));
	  // = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	  noise.y = make_real2(saru.gf(0, complex_gaussian_sc));
	  // = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	  noise.z = make_real2(saru.gf(0, complex_gaussian_sc));
	}
	/*Beware of nyquist points! They only appear with even cell dimensions
	  There are 8 nyquist points at most (cell=0,0,0 is excluded at the start of the kernel)
	  These are the 8 vertex of the inferior left cuadrant. The O points:
               +--------+--------+
              /|       /|       /|
             / |      / |      / | 
            +--------+--------+  |
           /|  |    /|  |    /|  |
          / |  +---/-|--+---/-|--+
         +--------+--------+  |	/|
         |  |/ |  |  |/ |  |  |/ |
         |  O-----|--O-----|--+	 |
         | /|6 |  | /|7 |  | /|	 |
         |/ |  +--|/-|--+--|/-|--+
         O--------O--------+  |	/ 
         |5 |/    |4 |/    |  |/  
         |  O-----|--O-----|--+	  
     ^   | / 3    | / 2    | /  ^ 
     |   |/       |/       |/  /  
     kz  O--------O--------+  ky  
         kx ->     1
	*/
	/*Handle nyquist points*/

	bool nyquist;
	{ //Is the current wave number a nyquist point?
	  bool isXnyquist = (cell.x == grid.cellDim.x - cell.x) && (grid.cellDim.x%2 == 0);
	  bool isYnyquist = (cell.y == grid.cellDim.y - cell.y) && (grid.cellDim.y%2 == 0);
	  bool isZnyquist = (cell.z == grid.cellDim.z - cell.z) && (grid.cellDim.z%2 == 0);

	  nyquist =  (isXnyquist && cell.y==0   && cell.z==0)  or  //1
               	     (isXnyquist && isYnyquist  && cell.z==0)  or  //2
               	     (cell.x==0  && isYnyquist  && cell.z==0)  or  //3
               	     (isXnyquist && cell.y==0   && isZnyquist) or  //4
               	     (cell.x==0  && cell.y==0   && isZnyquist) or  //5
               	     (cell.x==0  && isYnyquist  && isZnyquist) or  //6
               	     (isXnyquist && isYnyquist  && isZnyquist);    //7
	}
	
	if(nyquist){
	  /*Nyquist points are their own conjugates, so they must be real.
	    ||r||^2 = <x^2> = ||Real{z}||^2 = <Real{z}^2>·sqrt(2) =  prefactor*/
	  constexpr real nqsc = real(1.41421356237310); //sqrt(2)
	  noise.x.x *= nqsc; noise.x.y = 0;
	  noise.y.x *= nqsc; noise.y.y = 0;
	  noise.z.x *= nqsc; noise.z.y = 0;
	}
	/*Z = sqrt(B)·(I-k^k)·dW*/
	{// Compute for v_k wave number
	  const real3 k = cellToWaveNumber(cell, grid.cellDim, grid.box.boxSize);

	  const real invk2 = real(1.0)/dot(k,k);
	  /*Get my scaling factor B, Fourier representation of FCM*/
	  const real B = invk2/vis;
	  const real Bsq = sqrt(B/real(ncells));
	  
	  cufftComplex3 factor = noise;
	  factor.x *= Bsq;
	  factor.y *= Bsq;
	  factor.z *= Bsq;	  
	  
	  gridVelsFourier[icell] += projectFourier(k, factor);
	}
	/*Compute for conjugate v_{N-k} if needed*/
	
	/*Take care of conjugate wave number -> v_{Nx-kx,Ny-ky, Nz-kz}*/
	/*The special cases k_i=0 do not have conjugates, a.i N-k = N which is not stored*/
	
	if(nyquist) return; //Nyquist points do not have conjugates

	/*Conjugates are stored only when kx == Nx/2 or kx=0*/	
	if(cell.x == grid.cellDim.x-cell.x or cell.x == 0){
	  /*The only case with x conjugates is when kx = Nx-kx or kx=0, so this line is not needed*/
	  //if(cell.x > 0) cell.x = grid.cellDim.x-cell.x;
	  /*k_i=N_i is not stored, so do not conjugate them, the necessary exclusions are at the start of the kernel*/
	  if(cell.y > 0) cell.y = grid.cellDim.y-cell.y;
	  if(cell.z > 0) cell.z = grid.cellDim.z-cell.z;
	  
	  icell = grid.getCellIndex(cell);
	  
	  const real3 k = cellToWaveNumber(cell, grid.cellDim, grid.box.boxSize);

	  const real invk2 = real(1.0)/dot(k,k);
	  /*Get my scaling factor B,  Fourier representation of FCM*/
	  const real B = invk2/vis;

	  const real Bsq = sqrt(B/real(ncells));	  
	  cufftComplex3 factor = noise;
	  /*v_{N-k} = v*_k, so the complex noise must be conjugated*/
	  factor.x.y *= real(-1.0);
	  factor.y.y *= real(-1.0);
	  factor.z.y *= real(-1.0);
	  
	  factor.x *= Bsq;
	  factor.y *= Bsq;
	  factor.z *= Bsq;
	  
	  gridVelsFourier[icell] += projectFourier(k, factor);
	}
      }
      
      /*Interpolates a quantity (i.e velocity) from its values in the grid to the particles.
	For that it uses a Gaussian kernel of the form f(r) = prefactor·exp(-tau·r^2)
	σ = dx*dy*dz; h^3 in [1]
	Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw = 
	= σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)

	Input: gridVels = FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	Output: Mv = σ·St·gridVels
	The first term is computed in forceFourier2Vel and the second in fourierBrownianNoise
      */
      template<class Kernel, typename vtype>
      __global__ void grid2ParticlesD(const real4 * __restrict__ pos,
				      vtype * __restrict__ Mv, /*Result (i.e Mw·F)*/
				      const real3 * __restrict__ gridVels, /*Values in the grid*/
				      int N, /*Number of particles*/
				      Grid grid, /*Grid information and methods*/				  
				      Kernel kernel){
	/*A thread per particle */
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=N) return;
	/*Get my particle and my cell*/
    
	const real3 pi = make_real3(pos[id]);
	const int3 celli = grid.getCell(pi);

	/*J = S^T = St = σ S*/    
	real dV = (grid.cellSize.x*grid.cellSize.y*grid.cellSize.z);

	real3  result = make_real3(0);

	int3 cellj;
	int x,y,z;
	/*Transform cell number to cell center position*/
	const real3 cellPosOffset = real(0.5)*(grid.cellSize-grid.box.boxSize);
	const int P = kernel.support/2;
	/*Transvers the Pth neighbour cells*/
	for(z=-P; z<=P; z++){
	  cellj.z = grid.pbc_cell_coord<2>(celli.z + z);
	  for(y=-P; y<=P; y++){
	    cellj.y = grid.pbc_cell_coord<1>(celli.y + y);
	    for(x=-P; x<=P; x++){
	      cellj.x = grid.pbc_cell_coord<0>(celli.x + x);
	      /*Get neighbour cell*/	  
	      const int jcell = grid.getCellIndex(cellj);

	      /*Compute distance to center*/
	      const real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize - cellPosOffset);
	      /*Interpolate cell value and sum*/
	      const real3 cellj_vel = make_real3(gridVels[jcell]);
	      result += (dV*kernel.delta(rij))*cellj_vel;
	    }
	  }
	}
	/*Write total to global memory*/
	Mv[id] += result;
      }

    }
    

    //Spreads the particle quantity v to the grid, AKA applies the operator S to i.e the force.
    template<typename vtype>
    void FCM::spreadParticles(vtype *v, cudaStream_t st){
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      real3* d_gridVels = (real3*)thrust::raw_pointer_cast(gridVelsFourier.data());
      
      sys->log<System::DEBUG2>("[BDHI::FCM] Particles to grid");
      Kernel kernel(grid.cellSize);
      //Launch a small block per particle      
      {
	int support = kernel.support;
	int threadsPerParticle = 64;
	int numberNeighbourCells = support*support*support;
	if(numberNeighbourCells < 64) threadsPerParticle = 32;
	
	FCM_ns::particles2GridD<<<numberParticles, threadsPerParticle, 0, st>>>
	  (pos.raw(), v, d_gridVels, numberParticles, grid, kernel);
      }

    }

    //Takes grid forcing and transforms it to grid velocity, also adds the fluctuations.
    //For that it solves the eq. \vec{v} = B·\vec{g}+fluct. -> \hat{\vec{v}} =  1/(\eta h^3) 1/k^2 \hat{P} · \hat{\vec{g}} + \hat{fluct}
    //Being \hat{P} = (I-\vec{k}\otimes \vec{k}) the projection onto the space of divergence free velocity fields.
    //v = FFTi·(B·FFTf·g + 1/√(h^3)·√B·dWw)
    //g is the fluid forcing which includes S·F and any other external force density on the fluid.
    void FCM::convolveFourier(cudaStream_t st){
      CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
      CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));

      auto d_gridVels = thrust::raw_pointer_cast(gridVelsFourier.data());
      auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());
            
      sys->log<System::DEBUG2>("[BDHI::FCM] Taking grid to wave space");
      {
	/*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
	auto cufftStatus =
	  cufftExecR2C(cufft_plan_forward,
		       (cufftReal*)d_gridVels,
		       (cufftComplex*)d_gridVelsFourier);
	if(cufftStatus != CUFFT_SUCCESS){
	  sys->log<System::CRITICAL>("[BDHI::FCM] Error in forward CUFFT");
	}
      }
      sys->log<System::DEBUG2>("[BDHI::FCM] Wave space convolution");
      {
	/*Scale the wave space grid forces, transforming in velocities -> B·FFT·S·F*/
	//Launch a 3D grid of threads, a thread per cell.
	//Only the second half of the cells in the innermost (x) coordinate need to be processed, the rest are redundant and not used by cufft.
      
	dim3 NthreadsCells = dim3(8,8,8);
	dim3 NblocksCells;
	{
	  int ncellsx = grid.cellDim.x/2+1;
	  NblocksCells.x= (ncellsx/NthreadsCells.x + ((ncellsx%NthreadsCells.x)?1:0));
	  NblocksCells.y= grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
	  NblocksCells.z= grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
	}


            
	FCM_ns::forceFourier2Vel<<<NblocksCells, NthreadsCells, 0, st>>>
	  ((cufftComplex3*) d_gridVelsFourier, //Input: FFT·S·F
	   (cufftComplex3*) d_gridVelsFourier, //Output: B·FFT·S·F
	   viscosity,
	   grid);
	/*Add the stochastic noise to the fourier velocities if T>0 -> 1/√σ·√B·dWw */
	if(temperature > real(0.0)){
	  sys->log<System::DEBUG2>("[BDHI::FCM] Wave space brownian noise");
	  static ullint counter = 0; //Seed the rng differently each call
	  counter++;
	  sys->log<System::DEBUG2>("[BDHI::FCM] Wave space brownian noise");
	  real dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
	  real prefactor = sqrt(2*temperature/(dt*dV)); //See eq. 53 in [1]
	  FCM_ns::fourierBrownianNoise<<<NblocksCells, NthreadsCells, 0, st>>>(
			//In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw 
			(cufftComplex3*)d_gridVelsFourier, 
			grid,
			prefactor, // 1/√σ· sqrt(2*T/dt),
			viscosity,
			seed, //Saru needs two seeds apart from thread id
			counter);
	}
      }
      sys->log<System::DEBUG2>("[BDHI::FCM] Going back to real space");
      {
	/*Take the fourier velocities back to real space ->  FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
	auto cufftStatus =
	  cufftExecC2R(cufft_plan_inverse,
		       (cufftComplex*)d_gridVelsFourier,
		       (cufftReal*)d_gridVels);
	if(cufftStatus != CUFFT_SUCCESS){
	  sys->log<System::CRITICAL>("[BDHI::FCM] Error in inverse CUFFT");
	}
      }
      CudaCheckError();

    }

    //Interpolates the velocities of the grid to the particle positions and adds it to an array
    //\vec{Mv} = h^3·St·\vec{v}
    void FCM::interpolateParticles(real3 *Mv, cudaStream_t st){
      sys->log<System::DEBUG2>("[BDHI::FCM] Grid to particles");	    
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      real3* d_gridVels = (real3*)thrust::raw_pointer_cast(gridVelsFourier.data());
      
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0); 

      /*Interpolate the real space velocities back to the particle positions ->
	Output: Mv = Mw·F + sqrt(2*T/dt)·√Mw·dWw = σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
      FCM_ns::grid2ParticlesD<<<Nblocks, Nthreads, 0, st>>>
	(pos.raw(), Mv, d_gridVels,
	 numberParticles, grid, Kernel(grid.cellSize));
    }

    /*Compute M·F and B·dW in Fourier space
      σ = dx*dy*dz; h^3 in [1]
      Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw = 
      = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
    */
    template<typename vtype>
    void FCM::Mdot_far(real3 *Mv, vtype *v, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::FCM] Computing MF wave space....");
      /*Clean gridVels*/
      {
	int ncells = grid.cellDim.x*grid.cellDim.y*grid.cellDim.z;
	int BLOCKSIZE = 128;
	int Nthreads = BLOCKSIZE<ncells?BLOCKSIZE:ncells;
	int Nblocks  =  ncells/Nthreads +  ((ncells%Nthreads!=0)?1:0); 

	sys->log<System::DEBUG2>("[BDHI::FCM] Setting vels to zero...");
	//Note that the same storage space is used for Fourier and real space
	//The real space is the only one that needs to be cleared.
	auto d_gridVels = (real3*)thrust::raw_pointer_cast(gridVelsFourier.data());
	fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(d_gridVels,
						  make_real3(0), ncells);
      }

      spreadParticles(v, st);
      convolveFourier(st);
      interpolateParticles(Mv, st);      
    }

    void FCM::computeMF(real3* MF, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::FCM] Computing MF....");
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      Mdot<real4>(MF, force.raw(), st);
    }

    void FCM::computeBdW(real3* BdW, cudaStream_t st){
      //This part is included in Fourier space when computing MF
    }

    void FCM::computeDivM(real3* divM, cudaStream_t st){}


    void FCM::finish_step(cudaStream_t st){
      sys->log<System::DEBUG2>("[BDHI::FCM] Finishing step");
 
    }
  }
}

/* Raul P. Pelaez  2018-2021

  This code implements the algorithm described in [1], using cuFFT to solve te
velocity in eq. 24 of [1] and compute the brownian fluctuations of eq. 30 in [1]
(it only needs two FFT's). It only includes the stokeslet terms.

  The operator terminology used in the comments (as well as the wave space part
of the algorithm) comes from [2], the PSE basic reference.

You can choose different Kernels by changing the "using Kernel" below. A bunch of them are available in FCM_kernels.cuh
  References:
  [1] Fluctuating force-coupling method for simulations of colloidal suspensions. Eric E. Keaveny. 2014.
  [2]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations. Fiore, Balboa, Donev and Swan. 2017.

Contributors:
Pablo Palacios - 2021: Introduce the torques functionality.
 */
#ifndef FCM_IMPL_CUH
#define FCM_IMPL_CUH
#include "Integrator/BDHI/BDHI.cuh"
#include"utils/quaternion.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftComplex3.cuh"
#include "utils/container.h"
#include "utils/Grid.cuh"
#include "misc/IBM.cuh"
#include "FCM_kernels.cuh"
#include "utils/cufftDebug.h"
#include"utils/debugTools.h"
#include"utils.cuh"
#include<chrono>
namespace uammd{
  namespace BDHI{

    template<class Kernel, class KernelTorque>
    class FCM_impl{
    public:
      //using Kernel = FCM_ns::Kernels::Gaussian;
      //using Kernel = FCM_ns::Kernels::BarnettMagland;
      //using Kernel = FCM_ns::Kernels::Peskin::threePoint;
      //using Kernel = FCM_ns::Kernels::Peskin::fourPoint;
      //using Kernel = FCM_ns::Kernels::GaussianFlexible::sixPoint;
      //using KernelTorque =  FCM_ns::Kernels::GaussianTorque;
      // using cufftComplex = cufftComplex_t<real>;
      // using cufftComplex3 = cufftComplex3_t<real>;

      struct Parameters: BDHI::Parameters{
	int3 cells = make_int3(-1, -1, -1); //Number of Fourier nodes in each direction
	uint seed = 0;
	std::shared_ptr<Kernel> kernel = nullptr;
	std::shared_ptr<KernelTorque> kernelTorque = nullptr;
      };

      FCM_impl(Parameters par):
	par(par),
	viscosity(par.viscosity),
	hydrodynamicRadius(par.hydrodynamicRadius),
	box(par.box){
	System::log<System::MESSAGE>("[BDHI::FCM] Initialized");
	if(box.boxSize.x == real(0.0) && box.boxSize.y == real(0.0) && box.boxSize.z == real(0.0)){
	  System::log<System::CRITICAL>("[BDHI::FCM] Box of size zero detected, cannot work without a box! (make sure a box parameter was passed)");
	}
	this->seed = par.seed;
	if(par.seed == 0){
	  auto now = std::chrono::steady_clock::now().time_since_epoch();
	  this->seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
	}
	initializeGrid(par);
	initializeKernel(par);
	initializeKernelTorque(par);
	printMessages(par);
	initCuFFT();
	CudaSafeCall(cudaDeviceSynchronize());
	CudaCheckError();
      }

      ~FCM_impl(){
	cudaDeviceSynchronize();
	cufftDestroy(cufft_plan_inverse);
	cufftDestroy(cufft_plan_forward);
      }

      real getHydrodynamicRadius(){
	return hydrodynamicRadius;
      }

      real getSelfMobility(){
	//O(a^8) accuracy. See Hashimoto 1959.
	//With a Gaussian this expression has a minimum deviation from measuraments of 7e-7*rh at L=64*rh.
	//The translational invariance of the hydrodynamic radius however decreases arbitrarily with the tolerance.
	//Seems that this deviation decreases with L, so probably is due to the correction below missing something.
	long double rh = this->getHydrodynamicRadius();
	long double L = box.boxSize.x;
	long double a = rh/L;
	long double a2= a*a; long double a3 = a2*a;
	long double c = 2.83729747948061947666591710460773907l;
	long double b = 0.19457l;
	long double a6pref = 16.0l*M_PIl*M_PIl/45.0l + 630.0L*b*b;
	return  1.0l/(6.0l*M_PIl*viscosity*rh)*(1.0l-c*a+(4.0l/3.0l)*M_PIl*a3-a6pref*a3*a3);
      }

      Box getBox(){
	return this->box;
      }

      //Computes the velocities and angular velocities given the forces and torques
      // If torques is a nullptr, the torque computation is skipped and the second output is empty
      std::pair<cached_vector<real3>, cached_vector<real3>>
      computeHydrodynamicDisplacements(real4* pos, real4* force, real4* torque,
				       int numberParticles, real temperature, real prefactor, cudaStream_t st);

    private:

      cudaStream_t st;
      uint seed;

      real viscosity;
      real hydrodynamicRadius;

      std::shared_ptr<Kernel> kernel;
      std::shared_ptr<KernelTorque> kernelTorque;

      Box box;
      Grid grid;

      cufftHandle cufft_plan_forward, cufft_plan_inverse;
      thrust::device_vector<char> cufftWorkArea;

      Parameters par;

      void initializeGrid(Parameters par){
	int3 cellDim;
	real h;
	if(par.cells.x<=0){
	  if(par.hydrodynamicRadius<=0)
	    System::log<System::CRITICAL>("[BDHI::FCM] I need an hydrodynamic radius if cell dimensions are not provided!");
	  h = Kernel::adviseGridSize(par.hydrodynamicRadius, par.tolerance);
	  cellDim = nextFFTWiseSize3D(make_int3(par.box.boxSize/h));
	}
	else{
	  cellDim = par.cells;
	}
	this->grid = Grid(box, cellDim);
      }

      void initializeKernel(Parameters par){
	real h = std::min({grid.cellSize.x, grid.cellSize.y, grid.cellSize.z});
	if(not par.kernel)
	  this->kernel = std::make_shared<Kernel>(h, par.tolerance);
	else
	  this->kernel = par.kernel;
	this->hydrodynamicRadius = kernel->fixHydrodynamicRadius(h, grid.cellSize.x);
      }

      void initializeKernelTorque(Parameters par){
	if(not par.kernelTorque){
	  real a = this->getHydrodynamicRadius();
	  real width = a/(pow(6*sqrt(M_PI), 1/3.));
	  real h = std::min({grid.cellSize.x, grid.cellSize.y, grid.cellSize.z});
	  this->kernelTorque = std::make_shared<KernelTorque>(width, h, par.tolerance);
	}
	else{
	  this->kernelTorque = par.kernelTorque;
	}
      }

      void printMessages(Parameters par){
	auto rh = this->getHydrodynamicRadius();
	auto M0 = this->getSelfMobility();
	System::log<System::MESSAGE>("[BDHI::FCM] Using kernel: %s", type_name<Kernel>().c_str());
	System::log<System::MESSAGE>("[BDHI::FCM] Closest possible hydrodynamic radius: %g (%g requested)", rh, par.hydrodynamicRadius);
	System::log<System::MESSAGE>("[BDHI::FCM] Self mobility: %g", (double)M0);
	if(box.boxSize.x != box.boxSize.y || box.boxSize.y != box.boxSize.z || box.boxSize.x != box.boxSize.z){
	  System::log<System::WARNING>("[BDHI::FCM] Self mobility will be different for non cubic boxes!");
	}
	System::log<System::MESSAGE>("[BDHI::FCM] Box Size: %g %g %g", grid.box.boxSize.x, grid.box.boxSize.y, grid.box.boxSize.z);
	System::log<System::MESSAGE>("[BDHI::FCM] Grid dimensions: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
	System::log<System::MESSAGE>("[BDHI::FCM] Interpolation kernel support: %g rh max distance, %d cells total",
				     kernel->support*0.5*grid.cellSize.x/rh, kernel->support);
	System::log<System::MESSAGE>("[BDHI::FCM] h: %g %g %g", grid.cellSize.x, grid.cellSize.y, grid.cellSize.z);
	System::log<System::MESSAGE>("[BDHI::FCM] Requested kernel tolerance: %g", par.tolerance);
	if(kernel->support >= grid.cellDim.x or
	   kernel->support >= grid.cellDim.y or
	   kernel->support >= grid.cellDim.z){
	  System::log<System::ERROR>("[BDHI::FCM] Kernel support is too big, try lowering the tolerance or increasing the box size!.");
	}
      }

      void initCuFFT(){
	CufftSafeCall(cufftCreate(&cufft_plan_forward));
	CufftSafeCall(cufftCreate(&cufft_plan_inverse));
	CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
	CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));
	size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
	//This sizes have to be reversed according to the cufft docs
	int3 n = grid.cellDim;
	int3 cdtmp = {n.z, n.y, n.x};
	int3 inembed = {n.z, n.y, n.x};
	int3 oembed = {n.z, n.y, n.x/2+1};
	/*I want to make three 3D FFTs, each one using one of the three interleaved coordinates*/
	CufftSafeCall(cufftMakePlanMany(cufft_plan_forward,
					3, &cdtmp.x, /*Three dimensional FFT*/
					&inembed.x,
					/*Each FFT starts in 1+previous FFT index. FFTx in 0*/
					3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
					/*Same format in the output*/
					&oembed.x,
					3, 1,
					/*Perform 3 direct Batched FFTs*/
					CUFFT_Real2Complex<real>::value, 3,
					&cufftWorkSizef));
	System::log<System::DEBUG>("[BDHI::FCM] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
	/*Same as above, but with C2R for inverse FFT*/
	CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
					3, &cdtmp.x, /*Three dimensional FFT*/
					&oembed.x,
					/*Each FFT starts in 1+previous FFT index. FFTx in 0*/
					3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
					&inembed.x,
					3, 1,
					/*Perform 3 inverse batched FFTs*/
					CUFFT_Complex2Real<real>::value, 3,
					&cufftWorkSizei));
	size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei)+10;
	size_t free_mem, total_mem;
	CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
	System::log<System::DEBUG>("[BDHI::FCM] Necessary work space for cuFFT: %s, available: %s, total: %s",
				   printUtils::prettySize(cufftWorkSize).c_str(),
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(total_mem).c_str());
	if(free_mem<cufftWorkSize){
	  System::log<System::EXCEPTION>("[BDHI::FCM] Not enough memory in device to allocate cuFFT free %s, needed: %s!!",
					 printUtils::prettySize(free_mem).c_str(),
					 printUtils::prettySize(cufftWorkSize).c_str());
	  throw std::runtime_error("Not enough memory for cuFFT");
	}
	cufftWorkArea.resize(cufftWorkSize);
	auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
	CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
	CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
      }

    };

    namespace fcm_detail{

      struct ToReal3{
	template<class vtype>
	inline __device__ real3 operator()(vtype q){return make_real3(q);}
      };

      template<class IterPos, class IterForce, class Kernel>
      cached_vector<real3> spreadForces(IterPos& pos, IterForce& force,
					int numberParticles,
					std::shared_ptr<Kernel> kernel,
					Grid grid,
					cudaStream_t st){
      /*Spread force on particles to grid positions -> S·F*/
	System::log<System::DEBUG2>("[BDHI::FCM] Particles to grid");
	auto force_r3 = thrust::make_transform_iterator(force, ToReal3());
	int3 n = grid.cellDim;
	cached_vector<real3> gridVels(n.x*n.y*n.z);
	thrust::fill(thrust::cuda::par.on(st), gridVels.begin(), gridVels.end(), real3());
	auto d_gridVels = (real3*)thrust::raw_pointer_cast(gridVels.data());
	IBM<Kernel> ibm(kernel, grid);
	ibm.spread(pos, force_r3, d_gridVels, numberParticles, st);
	CudaCheckError();
	return gridVels;
      }

      cached_vector<cufftComplex3> forwardTransform(cached_vector<real3>& gridReal,
						    int3 n,
						    cufftHandle plan, cudaStream_t st){
	cached_vector<cufftComplex3> gridFourier((n.x/2+1)*n.y*n.z);
	thrust::fill(thrust::cuda::par.on(st), gridFourier.begin(), gridFourier.end(), cufftComplex3());
	auto d_gridFourier = (cufftComplex*) thrust::raw_pointer_cast(gridFourier.data());
	auto d_gridReal = (real*) thrust::raw_pointer_cast(gridReal.data());
	cufftSetStream(plan, st);
	/*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
	CufftSafeCall(cufftExecReal2Complex<real>(plan, d_gridReal, d_gridFourier));
	return gridFourier;
      }


      __global__ void addTorqueCurl(cufftComplex3 *gridTorquesFourier, cufftComplex3* gridVelsFourier, Grid grid){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	const int3 nk = grid.cellDim;
	if(id >= (nk.z*nk.y*(nk.x/2+1))) return;
	const int3 ik = indexToWaveNumber(id, nk);
	const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize);
	const real half = real(0.5);
	const bool isUnpairedX = ik.x == (nk.x - ik.x);
	const bool isUnpairedY = ik.y == (nk.y - ik.y);
	const bool isUnpairedZ = ik.z == (nk.z - ik.z);
	real Dx = isUnpairedX?0:k.x;
	real Dy = isUnpairedY?0:k.y;
	real Dz = isUnpairedZ?0:k.z;
	auto gridi = gridTorquesFourier[id];
	cufftComplex3 gridVeli;
	gridVeli.x = {half*(-Dy*gridi.z.y + Dz*gridi.y.y),
	  half*(Dy*gridi.z.x - Dz*gridi.y.x)};
	gridVeli.y = {half*(-Dz*gridi.x.y + Dx*gridi.z.y),
	  half*(Dz*gridi.x.x-Dx*gridi.z.x)};
	gridVeli.z = {half*(-Dx*gridi.y.y + Dy*gridi.x.y),
	  half*(Dx*gridi.y.x - Dy*gridi.x.x)};
	gridVelsFourier[id] += gridVeli;
      }


      template<class IterPos, class IterTorque, class Kernel>
      void addSpreadTorquesFourier(IterPos& pos, IterTorque& torque, int numberParticles,
				   Grid grid,
				   std::shared_ptr<Kernel> kernel,
				   cufftHandle plan,
				   cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st){
	/*Spread force on particles to grid positions -> S·F*/
	System::log<System::DEBUG2>("[BDHI::FCM] Spreading torques");
	int3 n = grid.cellDim;
	auto torque_r3 = thrust::make_transform_iterator(torque, ToReal3());
	cached_vector<real3> gridTorques(n.x*n.y*n.z);
	auto d_gridTorques3 = thrust::raw_pointer_cast(gridTorques.data());
	thrust::fill(thrust::cuda::par.on(st), gridTorques.begin(), gridTorques.end(), real3());
	IBM<Kernel> ibm(kernel, grid);
	ibm.spread(pos, torque_r3, d_gridTorques3, numberParticles, st);
	auto gridTorquesFourier = forwardTransform(gridTorques, grid.cellDim, plan, st);
	int BLOCKSIZE = 128;
	int numberCells = n.z*n.y*(n.x/2+1);
	uint Nthreads = BLOCKSIZE<numberCells?BLOCKSIZE:numberCells;
	uint Nblocks = numberCells/Nthreads +  ((numberCells%Nthreads!=0)?1:0);
	auto d_gridVelsFourier = (cufftComplex3*) thrust::raw_pointer_cast(gridVelsFourier.data());
	auto d_gridTorquesFourier3 = (cufftComplex3*) thrust::raw_pointer_cast(gridTorquesFourier.data());
	addTorqueCurl<<<Nblocks, Nthreads,0,st>>>(d_gridTorquesFourier3,d_gridVelsFourier, grid);
	CudaCheckError();
      }

      /*Scales fourier transformed forces in the regular grid to obtain velocities,
	(Mw·F)_deterministic = σ·St·FFTi·B·FFTf·S·F
	 Input: gridForces = FFTf·S·F
	 Output:gridVels = B·FFTf·S·F -> B \propto (I-k^k/|k|^2)
       */
      /*A thread per fourier node*/
      __global__ void forceFourier2Vel(const cufftComplex3 * gridForces, /*Input array*/
				       cufftComplex3 * gridVels, /*Output array, can be the same as input*/
				       real vis,
				       Grid grid/*Grid information and methods*/
				       ){
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id == 0){
	  gridVels[0] = cufftComplex3();
	  return;
	}
	const int3 ncells = grid.cellDim;
	if(id>=(ncells.z*ncells.y*(ncells.x/2+1))) return;
	const int3 waveNumber = indexToWaveNumber(id, ncells);
	const real3 k = waveNumberToWaveVector(waveNumber, grid.box.boxSize);
	const real invk2 = real(1.0)/dot(k,k);
	/*Get my scaling factor B, Fourier representation of FCM kernel*/
	const real B = invk2/(vis*real(ncells.x*ncells.y*ncells.z));
	cufftComplex3 factor = gridForces[id];
	factor.x *= B;
	factor.y *= B;
	factor.z *= B;
	gridVels[id] = projectFourier(k, factor);
      }

      void convolveFourier(cached_vector<cufftComplex3>& gridVelsFourier, real viscosity, Grid grid, cudaStream_t st){
	System::log<System::DEBUG2>("[BDHI::FCM] Wave space velocity scaling");
	/*Scale the wave space grid forces, transforming in velocities -> B·FFT·S·F*/
	auto d_gridVelsFourier = (cufftComplex3*) thrust::raw_pointer_cast(gridVelsFourier.data());
	const int3 n = grid.cellDim;
	int Nthreads = 128;
	int Nblocks = (n.z*n.y*(n.x/2+1))/Nthreads +1;
	forceFourier2Vel<<<Nblocks, Nthreads, 0, st>>> (d_gridVelsFourier, d_gridVelsFourier, viscosity, grid);
	CudaCheckError();
      }

      /*Computes the long range stochastic velocity term
	Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
	= σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	See sec. B.2 in [1]
	This kernel gets v_k = gridVelsFourier = B·FFtt·S·F as input and adds 1/√σ·√B(k)·dWw.
	Keeping special care that v_k = v*_{N-k}, which implies that dWw_k = dWw*_{N-k}
      */
      __global__ void fourierBrownianNoise(cufftComplex3 * gridVelsFourier,
					   Grid grid,
					   real prefactor,/* sqrt(2·T/dt)*/
					   real viscosity,
					   uint seed1,
					   uint seed2){
	const uint id = blockIdx.x*blockDim.x + threadIdx.x;
	const int3 nk = grid.cellDim;
	if(id >= (nk.z*nk.y*(nk.x/2+1))) return;
	const int3 cell = make_int3(id%(nk.x/2+1), (id/(nk.x/2+1))%nk.y, id/((nk.x/2+1)*nk.y));
	/*cuFFT R2C and C2R only store half of the innermost dimension, the one that varies the fastest
	  The input of R2C is real and the output of C2R is real.
	  The only way for this to be true is if v_k={i,j,k} = v*_k{N-i, N-j, N-k}
	  So the conjugates are redundant and the is no need to compute them nor store them except on two exceptions.
	  In this scheme, the only cases in which v_k and v_{N-k} are stored are:
	  1- When the innermost dimension coordinate is 0.
	  2- When the innermost dimension coordinate is N/2 and N is even.
	*/
	/*K=0 is not added, no stochastic motion is added to the center of mass*/
	if(id == 0 or
	   /*These terms will be computed along its conjugates*/
	   /*These are special because the conjugate of k_i=0 is k_i=N_i,
	     which is not stored and therfore must not be computed*/
	   (cell.x==0 and cell.y == 0 and 2*cell.z >= nk.z+1) or
	   (cell.x==0 and 2*cell.y >= nk.y + 1)) return;
	cufftComplex3 noise = generateNoise(prefactor, id, seed1, seed2);
	const bool nyquist =  isNyquistWaveNumber(cell, nk);
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
	  const int3 ik = indexToWaveNumber(id, nk);
	  const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize);
	  const real invk2 = real(1.0)/dot(k,k);
	  /*Get my scaling factor B,  Fourier representation of FCM*/
	  const real B = invk2/viscosity;
	  const real Bsq = sqrt(B);
	  cufftComplex3 factor = noise;
	  factor.x *= Bsq;
	  factor.y *= Bsq;
	  factor.z *= Bsq;
	  gridVelsFourier[id] += projectFourier(k, factor);
	}
	/*Compute for conjugate v_{N-k} if needed*/
	/*Take care of conjugate wave number -> v_{Nx-kx,Ny-ky, Nz-kz}*/
	/*The special cases k_i=0 do not have conjugates, a.i N-k = N which is not stored*/
	if(nyquist) return; //Nyquist points do not have conjugates
	/*Conjugates are stored only when kx == Nx/2 or kx=0*/
	if(cell.x == nk.x - cell.x or cell.x == 0){
	  int xc = cell.x;
	  int yc = (cell.y > 0)*(nk.y - cell.y);
	  int zc = (cell.z > 0)*(nk.z - cell.z);
	  int id_conj =  xc + (nk.x/2 + 1)*(yc + zc*nk.y);
	  const int3 ik = indexToWaveNumber(id, nk);
	  const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize);
	  const real invk2 = real(1.0)/dot(k,k);
	  /*Get my scaling factor B,  Fourier representation of FCM*/
	  const real B = invk2/viscosity;
	  const real Bsq = sqrt(B);
	  cufftComplex3 factor = noise;
	  /*v_{N-k} = v*_k, so the complex noise must be conjugated*/
	  factor.x.y *= real(-1.0);
	  factor.y.y *= real(-1.0);
	  factor.z.y *= real(-1.0);
	  factor.x *= Bsq;
	  factor.y *= Bsq;
	  factor.z *= Bsq;
	  gridVelsFourier[id_conj] += projectFourier(k, factor);
	}
}

      void addBrownianNoise(cached_vector<cufftComplex3>& gridVelsFourier,
			    real temperature, real viscosity, real prefactor,
			    uint seed,
			    Grid grid, cudaStream_t st){
	static uint seed2 = 0;
	//The sqrt(2*T/dt) factor needs to be here because far noise is summed to the M·F term.
	/*Add the stochastic noise to the fourier velocities if T>0 -> 1/√σ·√B·dWw */
	if(temperature > real(0.0)){
	  seed2++;
	  auto d_gridVelsFourier =  (cufftComplex3*)thrust::raw_pointer_cast(gridVelsFourier.data());
	  System::log<System::DEBUG2>("[BDHI::FCM] Wave space brownian noise");
	  const int3 n = grid.cellDim;
	  const real dV = grid.getCellVolume();
	  const real fourierNormalization = 1.0/(double(n.x)*n.y*n.z);
	  real noisePrefactor = prefactor*sqrt(fourierNormalization*2*temperature/(dV));
	  int Nthreads = 128;
	  int Nblocks = (n.z*n.y*(n.x/2+1))/Nthreads +1;
	  //In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw
	  fourierBrownianNoise<<<Nblocks, Nthreads, 0, st>>>(d_gridVelsFourier, grid,
							     noisePrefactor, // 1/√σ· sqrt(2*T/dt),
							     viscosity,
							     seed,
							     seed2);
	  CudaCheckError();
	}
      }

      cached_vector<real3> inverseTransform(cached_vector<cufftComplex3>& gridFourier,
					    int3 n, cufftHandle plan, cudaStream_t st){
	cached_vector<real3> gridReal(n.x*n.y*n.z);
	thrust::fill(thrust::cuda::par.on(st), gridReal.begin(), gridReal.end(), real3());
	auto d_gridFourier = (cufftComplex*) thrust::raw_pointer_cast(gridFourier.data());
	auto d_gridReal = (real*) thrust::raw_pointer_cast(gridReal.data());
	cufftSetStream(plan, st);
	//Take the grid fourier forces and apply take it to real space -> FFTf·S·F
	CufftSafeCall(cufftExecComplex2Real<real>(plan, d_gridFourier, d_gridReal));
	return gridReal;
      }

      template<class IterPos, class Kernel>
      cached_vector<real3> interpolateVelocity(IterPos& pos, cached_vector<real3>& gridVels,
					       Grid grid, std::shared_ptr<Kernel> kernel,
					       int numberParticles, cudaStream_t st){
	System::log<System::DEBUG2>("[BDHI::FCM] Grid to particles");
	/*Interpolate the real space velocities back to the particle positions ->
	  Output: Mv = Mw·F + sqrt(2*T/dt)·√Mw·dWw = σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
	real3* d_gridVels = thrust::raw_pointer_cast(gridVels.data());
	IBM<Kernel> ibm(kernel, grid);
	cached_vector<real3> linearVelocities(numberParticles);
	thrust::fill(thrust::cuda::par.on(st), linearVelocities.begin(), linearVelocities.end(), real3());
	auto d_linearVelocities = thrust::raw_pointer_cast(linearVelocities.data());
	ibm.gather(pos, d_linearVelocities, d_gridVels, numberParticles, st);
	CudaCheckError();
	return linearVelocities;
      }

      //Compute the curl of the velocity, V, in Fourier space. This is equal to the angular velocity
      // 0.5\nabla \times V = 0.5 (i*k_x i*k_y i*k_z)\times (V_x V_y V_z) =
      // = 0.5( i*k_y*V_z - i*k_z(V_y), i*k_z(V_x) - i*k_x*V_z, i*k_x*V_y - i*k_y*V_x)
      //Overwrite the output vector with the angular velocities in Fourier space
      //The input velocity vector is overwritten
      __global__ void computeVelocityCurlFourier(cufftComplex3 *gridVelsFourier, cufftComplex3* gridAngVelsFourier, Grid grid){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	const int3 nk = grid.cellDim;
	if(id >= nk.z*nk.y*(nk.x/2+1)){
	  gridAngVelsFourier[0] = cufftComplex3();
	  return;
	}
	const int3 cell = make_int3(id%(nk.x/2+1), (id/(nk.x/2+1))%nk.y, id/((nk.x/2+1)*nk.y));
	const int3 ik = indexToWaveNumber(id, nk);
	const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize);
	const real half = real(0.5);
	const bool isUnpairedX = ik.x == (nk.x - ik.x);
	const bool isUnpairedY = ik.y == (nk.y - ik.y);
	const bool isUnpairedZ = ik.z == (nk.z - ik.z);
	const real Dx = isUnpairedX?0:k.x;
	const real Dy = isUnpairedY?0:k.y;
	const real Dz = isUnpairedZ?0:k.z;
	cufftComplex3 gridAng;
	auto gridLinear = gridVelsFourier[id];
	gridAng.x = {half*(-Dy*gridLinear.z.y+Dz*gridLinear.y.y),
	  half*(Dy*gridLinear.z.x-Dz*gridLinear.y.x)};
	gridAng.y = {half*(-Dz*gridLinear.x.y+Dx*gridLinear.z.y),
	  half*(Dz*gridLinear.x.x-Dx*gridLinear.z.x)};
	gridAng.z = {half*(-Dx*gridLinear.y.y + Dy*gridLinear.x.y),
	  half*(Dx*gridLinear.y.x - Dy*gridLinear.x.x)};
	gridAngVelsFourier[id] = gridAng;
      }

      cached_vector<cufftComplex3> computeGridAngularVelocityFourier(cached_vector<cufftComplex3>& gridVelsFourier,
								     Grid grid,  cudaStream_t st){
	const int3 n = grid.cellDim;
	const int blockSize = 128;
	const int ncells = n.z*n.y*(n.x/2+1);
	const int numberBlocks = ncells/blockSize+1;
	auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());
	cached_vector<cufftComplex3> gridAngVelsFourier(gridVelsFourier.size());
	auto d_gridAngVelsFourier =  thrust::raw_pointer_cast(gridAngVelsFourier.data());
	computeVelocityCurlFourier<<<numberBlocks, blockSize, 0, st>>>(d_gridVelsFourier, d_gridAngVelsFourier, grid);
	CudaCheckError();
	return gridAngVelsFourier;
      }

      template<class IterPos, class Kernel>
      cached_vector<real3> interpolateAngularVelocity(IterPos& pos, cached_vector<real3>& gridAngVels,
						      Grid grid,
						      std::shared_ptr<Kernel> kernel,
						      int numberParticles, cudaStream_t st){
	IBM<Kernel> ibm(kernel, grid);
	cached_vector<real3> angularVelocities(numberParticles);
	thrust::fill(thrust::cuda::par.on(st), angularVelocities.begin(), angularVelocities.end(), real3());
	auto d_angularVelocities = thrust::raw_pointer_cast(angularVelocities.data());
	auto d_gridAngVels = thrust::raw_pointer_cast(gridAngVels.data());
	ibm.gather(pos, d_angularVelocities, d_gridAngVels, numberParticles, st);
	CudaCheckError();
	return angularVelocities;
      }
    }

    template<class Kernel, class KernelTorque>
    std::pair<cached_vector<real3>, cached_vector<real3>>
    FCM_impl<Kernel, KernelTorque>::computeHydrodynamicDisplacements(real4* pos, real4* force, real4* torque,
								     int numberParticles, real temperature, real prefactor, cudaStream_t st){
      using namespace fcm_detail;
      cached_vector<cufftComplex3> gridVelsFourier;
      if(force){
	auto gridVels = spreadForces(pos, force, numberParticles, kernel, grid, st);
	gridVelsFourier = forwardTransform(gridVels, grid.cellDim, cufft_plan_forward, st);
      }
      else{
	const auto n = grid.cellDim;
	gridVelsFourier.resize((n.x/2+1)*n.y*n.z);
	thrust::fill(thrust::cuda::par.on(st), gridVelsFourier.begin(), gridVelsFourier.end(), cufftComplex3());
      }
      if(torque){
	addSpreadTorquesFourier(pos, torque, numberParticles, grid, kernelTorque,
				cufft_plan_forward, gridVelsFourier, st);
      }
      if(force or torque){
	convolveFourier(gridVelsFourier, viscosity, grid, st);
      }
      addBrownianNoise(gridVelsFourier, temperature, viscosity, prefactor, seed, grid, st);
      cached_vector<real3> angularVelocities;
      if (torque){
	auto gridVelsFourier2 = gridVelsFourier;
	auto gridAngVelFourier = computeGridAngularVelocityFourier(gridVelsFourier2, grid, st);
	auto gridAngVel = inverseTransform(gridAngVelFourier, grid.cellDim, cufft_plan_inverse, st);
	angularVelocities = interpolateAngularVelocity(pos, gridAngVel, grid,
						       kernelTorque, numberParticles, st);
      }
      auto gridVels = inverseTransform(gridVelsFourier, grid.cellDim, cufft_plan_inverse, st);
      auto linearVelocities = interpolateVelocity(pos, gridVels, grid, kernel, numberParticles, st);
      CudaCheckError();
      return {linearVelocities, angularVelocities};
    }
  }
}

#endif

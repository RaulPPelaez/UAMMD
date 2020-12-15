/*Raul P. Pelaez 2017-2020. Positively Split Edwald Rotne-Prager-Yamakawa Brownian Dynamics with Hydrodynamic interactions.

Far field

*/

#ifndef BDHI_PSE_FARFIELD_CUH
#define BDHI_PSE_FARFIELD_CUH
#include"uammd.cuh"
#include"utils.cuh"
#include"third_party/saruprng.cuh"
#include"utils/debugTools.h"
#include "utils/cufftComplex3.cuh"
#include"utils/Grid.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftDebug.h"
#include "misc/IBM.cuh"
namespace uammd{
  namespace BDHI{
    namespace pse_ns{

      class Kernel{
	real prefactor;
	real tau;
      public:
	int support;
	Kernel(int P, real width){
	  this->support = 2*P+1;
	  this->prefactor = cbrt(1.0/(width*width*width*pow(2.0*M_PI, 1.5)));
	  this->tau = -0.5/(width*width);
	}

	inline __device__ real phi(real r) const{
	  return prefactor*exp(tau*r*r);
	}

      };

      class FarField{
      public:
	using cufftReal = cufftReal_t<real>;
	using cufftComplex = cufftComplex_t<real>;
	using cufftComplex3 =  cufftComplex3_t<real>;
	FarField(Parameters par, std::shared_ptr<System> sys, std::shared_ptr<ParticleData> pd, std::shared_ptr<ParticleGroup> pg):
	  box(par.box),
	  temperature(par.temperature),
	  viscosity(par.viscosity),
	  hydrodynamicRadius(par.hydrodynamicRadius),
	  dt(par.dt),
	  psi(par.psi),
	  shearStrain(par.shearStrain),
	  sys(sys), pd(pd), pg(pg)
	{
	  this->seed = sys->rng().next32();
	  initializeGrid(par.tolerance);
	  const int3 n = grid.cellDim;
	  gridVelsFourier.resize(3*n.z*n.y*(n.x/2+1), cufftComplex());
	  gridVels.resize(n.z*n.y*n.x, real3());
	  initializeKernel(par.tolerance);
	  initializeFourierFactors();
	  initializeCuFFT();
	  CudaCheckError();
	  sys->log<System::MESSAGE>("[BDHI_PSE] Gaussian kernel support: %d", this->kernel->support);
	  sys->log<System::MESSAGE>("[BDHI::PSE] Box Size: %f %f %f", box.boxSize.x, box.boxSize.y, box.boxSize.z);
	  sys->log<System::MESSAGE>("[BDHI::PSE] Unitless splitting factor ξ·a: %f", psi*par.hydrodynamicRadius);
	  sys->log<System::MESSAGE>("[BDHI::PSE] Far range grid size: %d %d %d", n.x, n.y, n.z);
	}

	~FarField(){
	  cufftDestroy(cufft_plan_inverse);
	  cufftDestroy(cufft_plan_forward);
	}

	void Mdot(real3 *Mv, cudaStream_t st);

      private:
	//template<class T> using gpu_container = thrust::device_vector<T, managed_allocator<T>>;
	//template<class T> using gpu_cached_container = thrust::device_vector<T, System::allocator_thrust<T>>;
	template<class T> using gpu_container = thrust::device_vector<T>;
	shared_ptr<ParticleData> pd;
	shared_ptr<ParticleGroup> pg;
	shared_ptr<System> sys;
	Box box;
	uint seed;

	void initializeCuFFT();
	void initializeKernel(real tolerance);
	void initializeFourierFactors();
	void initializeGrid(real tolerance);

	void spreadForce(cudaStream_t st);
	void forwardTransformForces(cudaStream_t st);
	void convolveFourier(cudaStream_t st);
	void addBrownianNoise(cudaStream_t st);
	void inverseTransformVelocity(cudaStream_t st);
	void interpolateVelocity(real3* MF, cudaStream_t st);

	real hydrodynamicRadius;
	real temperature;
	real viscosity;
	real dt;
	real psi; /*Splitting factor*/
	Grid grid;

	real eta; // kernel width
	real shearStrain;
	cufftHandle cufft_plan_forward, cufft_plan_inverse;
	gpu_container<char> cufftWorkArea;
	gpu_container<cufftComplex> gridVelsFourier;
	gpu_container<real3> gridVels;
	gpu_container<real> fourierFactor;
	std::shared_ptr<Kernel> kernel;
      };

      namespace detail{
	using cufftComplex3 = FarField::cufftComplex3;
	/*Apply the projection operator to a wave number with a certain complex factor.
	  res = (I-k^k)·factor
	  See i.e eq. 16 in [1].
	*/
	__device__ cufftComplex3 projectFourier(real3 k, cufftComplex3 factor){
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

	/*Scales fourier transformed forces in the regular grid to obtain velocities,
	  (Mw·F)_deterministic = σ·St·FFTi·B·FFTf·S·F
	  also adds stochastic fourier noise, see addBrownianNoise
	  Input: gridForces = FFTf·S·F
	  Output:gridVels = B·FFTf·S·F + 1/√σ·√B·dWw
	  See sec. B.2 in [1]
	*/
	__global__ void forceFourier2Vel(cufftComplex3 * gridForces, /*Input array*/
					 cufftComplex3 * gridVels, /*Output array, can be the same as input*/
					 /*Fourier scaling factors, see fillFourierScaling Factors*/
					 const real* Bfactor,
					 real shearStrain,
					 Grid grid){
	  const int id = blockIdx.x*blockDim.x + threadIdx.x;
	  if(id == 0){
	    gridVels[0] = cufftComplex3();
	    return;
	  }
	  const int3 ncells = grid.cellDim;
	  if(id>=(ncells.z*ncells.y*(ncells.x/2+1))) return;
	  const int3 waveNumber = indexToWaveNumber(id, ncells);
	  const real3 k = waveNumberToWaveVector(waveNumber, grid.box.boxSize, shearStrain);
	  const real B = Bfactor[id];
	  const cufftComplex3 factor = gridForces[id]*B;
	  gridVels[id] = projectFourier(k, factor);
	}

	/*Compute gaussian complex noise dW, std = prefactor -> ||z||^2 = <x^2>/sqrt(2)+<y^2>/sqrt(2) = prefactor*/
	/*A complex random number for each direction*/
	__device__ cufftComplex3 generateNoise(real prefactor, uint id, uint seed1, uint seed2){
	  //Uncomment to use uniform numbers instead of gaussian
	  Saru saru(id, seed1, seed2);
	  cufftComplex3 noise;
	  const real complex_gaussian_sc = real(0.707106781186547)*prefactor; //1/sqrt(2)
	  //const real sqrt32 = real(1.22474487139159)*prefactor;
	  // = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	  noise.x = make_real2(saru.gf(0, complex_gaussian_sc));
	  // = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	  noise.y = make_real2(saru.gf(0, complex_gaussian_sc));
	  // = make_real2(saru.f(-1.0f, 1.0f),saru.f(-1.0f, 1.0f))*sqrt32;
	  noise.z = make_real2(saru.gf(0, complex_gaussian_sc));
	  return noise;
	}

	__device__ bool isNyquistWaveNumber(int3 cell, int3 ncells){
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
	  //Is the current wave number a nyquist point?
	  const bool isXnyquist = (cell.x == ncells.x - cell.x) and (ncells.x%2 == 0);
	  const bool isYnyquist = (cell.y == ncells.y - cell.y) and (ncells.y%2 == 0);
	  const bool isZnyquist = (cell.z == ncells.z - cell.z) and (ncells.z%2 == 0);
	  const bool nyquist =  (isXnyquist and cell.y==0    and cell.z==0)  or  //1
	    (isXnyquist and isYnyquist and cell.z==0)  or  //2
	    (cell.x==0    and isYnyquist and cell.z==0)  or  //3
	    (isXnyquist and cell.y==0    and isZnyquist) or  //4
	    (cell.x==0    and cell.y==0    and isZnyquist) or  //5
	    (cell.x==0    and isYnyquist and isZnyquist) or  //6
	    (isXnyquist and isYnyquist and isZnyquist);    //7
	  return nyquist;
	}

	/*Computes the long range stochastic velocity term
	Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
	= σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	See sec. B.2 in [1]
	This kernel gets v_k = gridVelsFourier = B·FFtt·S·F as input and adds 1/√σ·√B(k)·dWw.
	Keeping special care that v_k = v*_{N-k}, which implies that dWw_k = dWw*_{N-k}
	*/
	__global__ void fourierBrownianNoise(cufftComplex3 *__restrict__ gridVelsFourier,
					     const real* __restrict__ Bfactor,
					     Grid grid,
					     real prefactor,/* sqrt(2·T/dt)*/
					     real shearStrain,
					     uint seed1, uint seed2){
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
	    const real Bsq = sqrt(Bfactor[id]);
	    cufftComplex3 factor = noise;
	    const int3 ik = indexToWaveNumber(id, nk);
	    const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize, shearStrain);
	    factor *= Bsq;
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
	    const int3 ik = indexToWaveNumber(id_conj, nk);
	    const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize, shearStrain);
	    const real Bsq = sqrt(Bfactor[id_conj]);
	    cufftComplex3 factor = noise;
	    /*v_{N-k} = v*_k, so the complex noise must be conjugated*/
	    factor.x.y *= real(-1.0);
	    factor.y.y *= real(-1.0);
	    factor.z.y *= real(-1.0);
	    factor *= Bsq;
	    gridVelsFourier[id_conj] += projectFourier(k, factor);
	  }
	}

	struct ToReal3{
	  template<class T>
	  __device__ real3 operator()(T vec){
	    return make_real3(vec);
	  }
	};
      }

      void FarField::spreadForce(cudaStream_t st){
	/*Spread force on particles to grid positions -> S·F*/
	sys->log<System::DEBUG2>("[BDHI::PSE] Particles to grid");
	int numberParticles = pg->getNumberParticles();
	auto force = pd->getForce(access::location::gpu, access::mode::read);
	auto force_r3 = thrust::make_transform_iterator(force.begin(), detail::ToReal3());
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto d_gridVels = (real3*)thrust::raw_pointer_cast(gridVels.data());
	IBM<Kernel> ibm(sys, kernel, grid);
	ibm.spread(pos.begin(), force_r3, d_gridVels, numberParticles, st);
	CudaCheckError();
      }

      void FarField::forwardTransformForces(cudaStream_t st){
	sys->log<System::DEBUG2>("[BDHI::PSE] Taking grid to wave space");
	auto d_gridVels = (real*)thrust::raw_pointer_cast(gridVels.data());
	auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());
	cufftSetStream(cufft_plan_forward, st);
	/*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
	CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, d_gridVels, d_gridVelsFourier));
      }

      void FarField::convolveFourier(cudaStream_t st){
	sys->log<System::DEBUG2>("[BDHI::PSE] Wave space velocity scaling");
	/*Scale the wave space grid forces, transforming in velocities -> B·FFT·S·F*/
	auto d_gridVelsFourier = (cufftComplex3*) thrust::raw_pointer_cast(gridVelsFourier.data());
	auto d_fourierFactor = thrust::raw_pointer_cast(fourierFactor.data()); 	//B in [1]
	const int3 n = grid.cellDim;
	int Nthreads = 128;
	int Nblocks = (n.z*n.y*(n.x/2+1))/Nthreads +1;
	detail::forceFourier2Vel<<<Nblocks, Nthreads, 0, st>>> (d_gridVelsFourier, d_gridVelsFourier, d_fourierFactor, shearStrain, grid);
	CudaCheckError();
      }

      void FarField::addBrownianNoise(cudaStream_t st){
	//eq 19 and beyond in [1].
	//The sqrt(2*T/dt) factor needs to be here because far noise is summed to the M·F term.
	/*Add the stochastic noise to the fourier velocities if T>0 -> 1/√σ·√B·dWw */
	if(temperature > real(0.0)){
	  auto d_gridVelsFourier = (cufftComplex3*) thrust::raw_pointer_cast(gridVelsFourier.data());
	  uint seed2 = sys->rng().next32();
	  sys->log<System::DEBUG2>("[BDHI::PSE] Wave space brownian noise");
	  auto d_fourierFactor = thrust::raw_pointer_cast(fourierFactor.data()); 	//B in [1]
	  const int3 n = grid.cellDim;
	  const real dV = grid.getCellVolume();
	  real prefactor = sqrt(2*temperature/(dt*dV));
	  int Nthreads = 128;
	  int Nblocks = (n.z*n.y*(n.x/2+1))/Nthreads +1;
	  //In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw
	  detail::fourierBrownianNoise<<<Nblocks, Nthreads, 0, st>>>(d_gridVelsFourier, d_fourierFactor, grid,
								     prefactor, // 1/√σ· sqrt(2*T/dt),
								     shearStrain,
								     seed, //Saru needs two seeds apart from thread id
								     seed2);
	}
      }

      void FarField::inverseTransformVelocity(cudaStream_t st){
        sys->log<System::DEBUG2>("[BDHI::PSE] Going back to real space");
	auto d_gridVels = (real*)thrust::raw_pointer_cast(gridVels.data());
	auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());
	cufftSetStream(cufft_plan_inverse, st);
	/*Take the fourier velocities back to real space ->  FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
	CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse,d_gridVelsFourier, d_gridVels));
      }

      void FarField::interpolateVelocity(real3* MF, cudaStream_t st){
	sys->log<System::DEBUG2>("[BDHI::PSE] Grid to particles");
	/*Interpolate the real space velocities back to the particle positions ->
	  Output: Mv = Mw·F + sqrt(2*T/dt)·√Mw·dWw = σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
	int numberParticles = pg->getNumberParticles();
	auto pos = pd->getPos(access::location::gpu, access::mode::read);
	auto d_gridVels = (real3*)thrust::raw_pointer_cast(gridVels.data());
	IBM<Kernel> ibm(sys, kernel, grid);
	ibm.gather(pos.begin(), MF, d_gridVels, numberParticles, st);
	CudaCheckError();
      }

    /*Far contribution of M·F and B·dW
      Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
      = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
    */
      void FarField::Mdot(real3 *MF, cudaStream_t st){
	sys->log<System::DEBUG1>("[BDHI::PSE] Computing MF wave space....");
	sys->log<System::DEBUG2>("[BDHI::PSE] Setting vels to zero...");
	thrust::fill(thrust::cuda::par.on(st), gridVels.begin(), gridVels.end(), real3());
	spreadForce(st);
	forwardTransformForces(st);
	convolveFourier(st);
	addBrownianNoise(st);
	inverseTransformVelocity(st);
	interpolateVelocity(MF, st);
	sys->log<System::DEBUG2>("[BDHI::PSE] MF wave space Done");
      }

      void FarField::initializeCuFFT(){
	CufftSafeCall(cufftCreate(&cufft_plan_forward));
	CufftSafeCall(cufftCreate(&cufft_plan_inverse));
	CufftSafeCall(cufftSetAutoAllocation(cufft_plan_forward, 0));
	CufftSafeCall(cufftSetAutoAllocation(cufft_plan_inverse, 0));
	size_t cufftWorkSizef = 0, cufftWorkSizei = 0;
	/*Set up cuFFT*/
	int3 n = grid.cellDim;
	int3 cdtmp = {n.z, n.y, n.x};
	int3 inembed = {n.z, n.y, n.x};//2*(n.x/2+1)};
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
	sys->log<System::DEBUG>("[BDHI::PSE] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
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
	size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei);
	sys->log<System::DEBUG>("[BDHI::PSE] Necessary work space for cuFFT: %s", printUtils::prettySize(cufftWorkSize).c_str());
	size_t free_mem, total_mem;
	CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
	if(free_mem<cufftWorkSize){
	  sys->log<System::EXCEPTION>("[BDHI::PSE] Not enough memory in device to allocate cuFFT free %s, needed: %s!!, try lowering the splitting parameter!",
				      printUtils::prettySize(free_mem).c_str(),
				      printUtils::prettySize(cufftWorkSize).c_str());
	  throw std::runtime_error("Not enough memory for cuFFT");
	}
	cufftWorkArea.resize(cufftWorkSize+1);
	auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
	CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
	CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
	CudaCheckError();
      }

      void FarField::initializeKernel(real tolerance){
	/*Grid spreading/interpolation parameters*/
	/*Gaussian spreading/interpolation kernel support points neighbour distance
	  See eq. 19 and sec 4.1 in [2]*/
	//m = C·sqrt(pi·P), from sec 4.1 we choose C=0.976
	constexpr double C = 0.976;
	double m = 1;
	/*I have empirically found that 0.1*tolerance here gives the desired overal accuracy*/
	while(erfc(m/sqrt(2)) > 0.1*tolerance) m+= 0.01;
	//This is P in [2]
	int support;
	//Support must be odd
	while( (support = int(pow(m/C, 2)/M_PI+0.5)+1 ) % 2 == 0) m+=tolerance;
	//P is in each direction, nearest neighbours -> P=1
        int P = support/2;
	//If P is too large for the grid set it to the grid size-1 (or next even number)
	const int3 n = grid.cellDim;
	int minCellDim = std::min({n.x, n.y, n.z});
	if(support>minCellDim){
	  support = minCellDim;
	  if(support%2==0) support--; //minCellDim will be 3 at least
	  P = support/2;
	  m = C*sqrt(M_PI*support);
	}
	double pw = 2*P+1;
	double h = std::min({grid.cellSize.x, grid.cellSize.y, grid.cellSize.z});
	/*Number of standard deviations in the grid's Gaussian kernel support*/
	double gaussM = m;
	/*Standard deviation of the Gaussian kernel*/
	double w   = pw*h/2.0;
	/*Gaussian splitting parameter*/
	this->eta = pow(2.0*psi*w/gaussM, 2);
	sys->log<System::MESSAGE>("[BDHI::PSE] eta: %g", eta);
	kernel = std::make_shared<Kernel>(P, sqrt(eta)/(2.0*psi));
      }

      namespace detail{
	/* Precomputes the fourier scaling factor B (see eq. 9 and 20.5 in [1]),
	 Bfactor = B(||k||^2, xi, tau) = 1/(vis·Vol) · sinc(k·rh)^2/k^2·Hashimoto(k,xi,tau)
      */
	__global__ void fillFourierScalingFactor(real * __restrict__ Bfactor,
						 Grid grid,
						 double rh, //Hydrodynamic radius						 
						 double viscosity,
						 double split,
						 double eta, //Gaussian kernel splitting parameter
						 double shearStrain
						 ){
	  const int id = blockIdx.x*blockDim.x + threadIdx.x;
	  const int3 ncells = grid.cellDim;
	  if(id>=(ncells.z*ncells.y*(ncells.x/2+1))) return;
	  if(id == 0){
	    Bfactor[0] = 0;
	    return;
	  }
	  const int3 waveNumber = indexToWaveNumber(id, ncells);
	  const real3 K = waveNumberToWaveVector(waveNumber, grid.box.boxSize, shearStrain);
	  /*Compute the scaling factor for this node*/
	  double k2 = dot(K,K);
	  double kmod = sqrt(k2);
	  double invk2 = 1.0/k2;
	  double sink = sin(kmod*rh);
	  double k2_invsplit2_4 = k2/(4.0*split*split);
	  /*The Hashimoto splitting function,
	    split is the splitting between near and far contributions,
	    eta is the splitting of the gaussian kernel used in the grid interpolation, see sec. 2 in [2]*/
	  /*See eq. 11 in [1] and eq. 11 and 14 in [2]*/
	  double tau = -k2_invsplit2_4*(1.0-eta);
	  double hashimoto = (1.0 + k2_invsplit2_4)*exp(tau)/k2;
	  /*eq. 20.5 in [1]*/
	  double B = sink*sink*invk2*hashimoto/(viscosity*rh*rh);
	  B /= double(ncells.x*ncells.y*ncells.z);
	  /*Store theresult in global memory*/
	  Bfactor[id] = B;
	}

      }

      void FarField::initializeFourierFactors(){
	/*B in [1], this array stores, for each cell/fourier node,
	  the scaling factor to go from forces to velocities in fourier space*/
	fourierFactor.resize(grid.cellDim.z*grid.cellDim.y*(grid.cellDim.x/2+1), real());
	int Nthreads = 128;
	int Nblocks = (grid.cellDim.z*grid.cellDim.y*(grid.cellDim.x/2+1))/Nthreads +1;
	detail::fillFourierScalingFactor<<<Nblocks, Nthreads>>>
	  (thrust::raw_pointer_cast(fourierFactor.data()), grid,
	   hydrodynamicRadius, viscosity, psi, eta, shearStrain);
      }

      void FarField::initializeGrid(real tolerance){
	real kcut = 2*psi*sqrt(-log(tolerance));
	sys->log<System::MESSAGE>("[BDHI::PSE] Far range wave number cut off: %f", kcut);
	const double hgrid = 2*M_PI/kcut;
	int3 cellDim = make_int3(2*box.boxSize/hgrid)+1;
	cellDim = nextFFTWiseSize3D(cellDim);
	this->grid = Grid(box, cellDim);
      }
    }
  }
}
#endif

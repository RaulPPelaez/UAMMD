/*Raul P. Pelaez 2017-2020. Positively Split Edwald Rotne-Prager-Yamakawa Brownian Dynamics with Hydrodynamic interactions.

Far field

Ondrej implemented the sheared cell functionality.
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
#include"utils/container.h"
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

	inline __device__ real phi(real r, real3 pos) const{
	  return prefactor*exp(tau*r*r);
	}

      };

      namespace detail{
	using cufftComplex3 = cufftComplex3_t<real>;
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

	/* Precomputes the fourier scaling factor B (see eq. 9 and 20.5 in [1]),
	   Returns B(||k||^2, xi, eta) = 1/(vis·Vol) · sinc(k·rh)^2/k^2·Hashimoto(k,xi,eta)
	*/
	__device__ real greensFunction(real3 waveVector,
				       real shearStrain,
				       real rh, //Hydrodynamic radius
				       real viscosity,
				       real split, //Ewald splitting
				       real eta, //Hasimoto Gaussian kernel splitting 
				       int3 n){
	  const real k2 = dot(waveVector, waveVector);
	  if(k2 == 0){
	    return real(0.0);
	  }
	  const real3 K_NUFFT = waveVector;
	  const real3 K_Ewald = shearWaveVector(waveVector, shearStrain);
	  /*Compute the scaling factor for this node*/
	  real K_Ewald2 = dot(K_Ewald,K_Ewald);
	  real K_NUFFT2 = dot(K_NUFFT,K_NUFFT);
	  real kmod = sqrt(K_Ewald2);
	  real invk2 = real(1.0)/K_Ewald2;
	  real sink = sin(kmod*rh);
	  real kEw2_invsplit2_4 = K_Ewald2/(real(4.0)*split*split);
	  real kNU2_invsplit2_4 = K_NUFFT2/(real(4.0)*split*split);
	  /*The Hashimoto splitting function,
	    split is the splitting between near and far contributions,
	    eta is the splitting of the gaussian kernel used in the grid interpolation, see sec. 2 in [2]*/
	  /*See eq. 11 in [1] and eq. 11 and 14 in [2]*/
	  /* Modification for shear strain: the right exponential is 
	     exp ((eta*kNUFFT^2- kEwald^2)/(4*xi^2)) */
	  real tau = eta*kNU2_invsplit2_4-kEw2_invsplit2_4;
	  real hashimoto = (real(1.0) + kEw2_invsplit2_4)*exp(tau)/K_Ewald2;
	  /*eq. 20.5 in [1]*/
	  real B = sink*sink*invk2*hashimoto/(viscosity*rh*rh);
	  B /= real(n.x*n.y*n.z);
	  //B(k)*(I- k^k/k^2)
	  return B;
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
					 real shearStrain,
					 real hydrodynamicRadius, real viscosity, real split, real eta,
					 Grid grid){
	  const int id = blockIdx.x*blockDim.x + threadIdx.x;
	  const int3 ncells = grid.cellDim;
	  if(id==0){
	    gridVels[0] = cufftComplex3();
	    return;
	  }
	  if(id>=(ncells.z*ncells.y*(ncells.x/2+1))) return;
	  const int3 waveNumber = indexToWaveNumber(id, ncells);
	  const real3 waveVector = waveNumberToWaveVector(waveNumber, grid.box.boxSize);
	  const real B = greensFunction(waveVector, shearStrain,
					hydrodynamicRadius, viscosity, split, eta, ncells);
	  gridVels[id] = projectFourier(shearWaveVector(waveVector, shearStrain),
					B*gridForces[id]);
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
	__global__ void fourierBrownianNoise(cufftComplex3 *gridVelsFourier,
					     Grid grid,
					     real prefactor,/* sqrt(2·T/dt)*/
					     real shearStrain,
					     real hydrodynamicRadius, real viscosity, real split, real eta,
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
	    const int3 ik = indexToWaveNumber(id, nk);
	    const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize);
	    const real B = greensFunction(k, shearStrain, hydrodynamicRadius, viscosity, split, eta, nk);
	    gridVelsFourier[id] += sqrt(B)*projectFourier(shearWaveVector(k, shearStrain), noise);
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
	    const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize);
	    cufftComplex3 factor = noise;
	    /*v_{N-k} = v*_k, so the complex noise must be conjugated*/
	    factor.x.y *= real(-1.0);
	    factor.y.y *= real(-1.0);
	    factor.z.y *= real(-1.0);
	    const real B = greensFunction(k, shearStrain, hydrodynamicRadius, viscosity, split, eta, nk);
	    gridVelsFourier[id_conj] += sqrt(B)*projectFourier(shearWaveVector(k, shearStrain), factor);
	  }
	}

	struct ToReal3{
	  template<class T>
	  __device__ real3 operator()(T vec){
	    return make_real3(vec);
	  }
	};
      }

      class FarField{
      public:
	using cufftReal = cufftReal_t<real>;
	using cufftComplex = cufftComplex_t<real>;
	using cufftComplex3 =  detail::cufftComplex3;

	FarField(std::shared_ptr<System> sys, Parameters par):
	  sys(sys),
	  box(par.box),
	  viscosity(par.viscosity),
	  hydrodynamicRadius(par.hydrodynamicRadius),
	  psi(par.psi){
	  this->seed = sys->rng().next32();
	  setShearStrain(par.shearStrain);
	  initializeGrid(par.tolerance);
	  initializeKernel(par.tolerance);
	  initializeCuFFT();
	  CudaCheckError();
	  System::log<System::MESSAGE>("[BDHI_PSE] Gaussian kernel support: %d", this->kernel->support);
	  System::log<System::MESSAGE>("[BDHI::PSE] Box Size: %f %f %f", box.boxSize.x, box.boxSize.y, box.boxSize.z);
	  System::log<System::MESSAGE>("[BDHI::PSE] Unitless splitting factor ξ·a: %f", psi*par.hydrodynamicRadius);
	  const int3 n = grid.cellDim;
	  System::log<System::MESSAGE>("[BDHI::PSE] Far range grid size: %d %d %d", n.x, n.y, n.z);
	}

	~FarField(){
	  cufftDestroy(cufft_plan_inverse);
	  cufftDestroy(cufft_plan_forward);
	}

	/*Far contribution of M·F and B·dW
	  Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·G_k·FFTf·S·F+ √σ·St·FFTi·√G_k·dWw =
	  = σ·St·FFTi( G_k·FFTf·S·F + 1/√σ·√G_k·dWw)
	*/
	void computeHydrodynamicDisplacements(real4* pos, real4* forces, real3 *Mv, int numberParticles,
					      real temperature, real prefactor, cudaStream_t st);

	void setShearStrain(real newStrain){
	  this->shearStrain = newStrain;
	}

      private:
	std::shared_ptr<System> sys;
	template<class T> using cached_vector = uninitialized_cached_vector<T>;
	template<class T> using gpu_container = thrust::device_vector<T>;
	Box box;
	uint seed;

	real shearStrain;
	real hydrodynamicRadius;
	real viscosity;
	real psi; /*Splitting factor*/
	Grid grid;

	real eta; // kernel width
	cufftHandle cufft_plan_forward, cufft_plan_inverse;
	size_t cufftWorkAreaSize;
	std::shared_ptr<Kernel> kernel;

	void initializeCuFFT();
	void initializeKernel(real tolerance);
	void initializeGrid(real tolerance);

	//Computes S·F
	auto spreadForce(real4* pos, real4* forces, int numberParticles, cudaStream_t st){
	  /*Spread force on particles to grid positions -> S·F*/
	  System::log<System::DEBUG2>("[BDHI::PSE] Particles to grid");
	  //auto force = pd->getForce(access::location::gpu, access::mode::read);
	  auto force_r3 = thrust::make_transform_iterator(forces, detail::ToReal3());
	  const int3 n = grid.cellDim;
	  cached_vector<real3> gridVels(n.x*n.y*n.z);
	  thrust::fill(thrust::cuda::par.on(st), gridVels.begin(), gridVels.end(), real3());
	  auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
	  IBM<Kernel> ibm(kernel, grid);
	  ibm.spread(pos, force_r3, d_gridVels, numberParticles, st);
	  CudaCheckError();
	  return gridVels;
	}

	//Applies the FFT operator
        auto forwardTransformForces(cached_vector<real3> &gridVels, cudaStream_t st){
	  System::log<System::DEBUG2>("[BDHI::PSE] Taking grid to wave space");
	  cached_vector<char> cufftWorkArea(cufftWorkAreaSize);
	  auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
	  CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea)); 
	  auto d_gridVels = (real*)thrust::raw_pointer_cast(gridVels.data());
	  const int3 n = grid.cellDim;
	  cached_vector<cufftComplex3> gridVelsFourier(n.z*n.y*(n.x/2+1));
	  auto d_gridVelsFourier = (cufftComplex*) thrust::raw_pointer_cast(gridVelsFourier.data());
	  cufftSetStream(cufft_plan_forward, st);
	  /*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
	  CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, d_gridVels, d_gridVelsFourier));
	  return gridVelsFourier;
	}

	//Multiplies by the Greens function in Fourier space (G_k)
	void convolveFourier(cached_vector<cufftComplex3> & gridVelsFourier, cudaStream_t st){
	  System::log<System::DEBUG2>("[BDHI::PSE] Wave space velocity scaling");
	  /*Scale the wave space grid forces, transforming in velocities -> B·FFT·S·F*/
	  auto d_gridVelsFourier = (cufftComplex3*) thrust::raw_pointer_cast(gridVelsFourier.data());
	  const int3 n = grid.cellDim;
	  int Nthreads = 128;
	  int Nblocks = (n.z*n.y*(n.x/2+1))/Nthreads +1;
	  detail::forceFourier2Vel<<<Nblocks, Nthreads, 0, st>>> (d_gridVelsFourier, d_gridVelsFourier,
								  shearStrain,
								  hydrodynamicRadius, viscosity, psi, eta,
								  grid);
	  CudaCheckError();
	}

	//Returns G_k·FFT(S·F)
	auto deterministicPart(real4* pos, real4* forces, int numberParticles, cudaStream_t st){
	  //In the absence of forces there is no need to compute the deterministic part
	  if(forces){
	    //S·F
	    auto gridVels = spreadForce(pos, forces, numberParticles, st);
	    // FFT(S·F)
	    auto gridVelsFourier = forwardTransformForces(gridVels, st);
	    // G_k·FFT(S·F)
	    convolveFourier(gridVelsFourier, st);
	    return gridVelsFourier;
	  }
	  else{
	    const int3 n = grid.cellDim;
	    cached_vector<cufftComplex3> gridVelsFourier(n.z*n.y*(n.x/2+1));
	    thrust::fill(thrust::cuda::par.on(st),
			 gridVelsFourier.begin(), gridVelsFourier.end(), cufftComplex3());
	    return gridVelsFourier;
	  }
	}

	//Adds the stochastic part in Fourier space ( 1/dV sqrt(G_k)·dW )
	void addBrownianNoise(cached_vector<cufftComplex3> & gridVelsFourier,
			      real temperature, real prefactor, cudaStream_t st){
	  //eq 19 and beyond in [1].
	  //The sqrt(2*T/dt) factor needs to be here because far noise is summed to the M·F term.
	  /*Add the stochastic noise to the fourier velocities if T>0 -> 1/√σ·√B·dWw */
	  if(temperature > real(0.0)){
	    auto d_gridVelsFourier = (cufftComplex3*) thrust::raw_pointer_cast(gridVelsFourier.data());
	    uint seed2 = sys->rng().next32();
	    System::log<System::DEBUG2>("[BDHI::PSE] Wave space brownian noise");
	    const int3 n = grid.cellDim;
	    const real dV = grid.getCellVolume();
	    real noise_prefactor = prefactor*sqrt(2*temperature/(dV));
	    int Nthreads = 128;
	    int Nblocks = (n.z*n.y*(n.x/2+1))/Nthreads +1;
	    //In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw
	    detail::fourierBrownianNoise<<<Nblocks, Nthreads, 0, st>>>(d_gridVelsFourier, grid,
								       noise_prefactor, // 1/√σ· sqrt(2*T/dt),
								       shearStrain,
								       hydrodynamicRadius, viscosity, psi, eta,
								       seed, //Saru needs two seeds apart from thread id
								       seed2);
	  }
	}

	//Applies the FFT^-1 operator
	auto inverseTransformVelocity(cached_vector<cufftComplex3> & gridVelsFourier, cudaStream_t st){
	  System::log<System::DEBUG2>("[BDHI::PSE] Going back to real space");
	  cached_vector<char> cufftWorkArea(cufftWorkAreaSize);
	  auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
	  CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
	  const int3 n = grid.cellDim;
	  cached_vector<real3> gridVels(n.x*n.y*n.z);
	  auto d_gridVels = (real*)thrust::raw_pointer_cast(gridVels.data());
	  auto d_gridVelsFourier = (cufftComplex*) thrust::raw_pointer_cast(gridVelsFourier.data());
	  cufftSetStream(cufft_plan_inverse, st);
	  /*Take the fourier velocities back to real space ->  FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
	  CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse,d_gridVelsFourier, d_gridVels));
	  return gridVels;
	}

	//Computes J·v. Interpolates the velocities at the particles positions
	void interpolateVelocity(cached_vector<real3> &gridVels, real4* pos, real3* MF,
				 int numberParticles, cudaStream_t st){
	System::log<System::DEBUG2>("[BDHI::PSE] Grid to particles");
	/*Interpolate the real space velocities back to the particle positions ->
	  Output: Mv = Mw·F + sqrt(2*T/dt)·√Mw·dWw = σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
	auto d_gridVels = (real3*)thrust::raw_pointer_cast(gridVels.data());
	IBM<Kernel> ibm(kernel, grid);
	ibm.gather(pos, MF, d_gridVels, numberParticles, st);
	CudaCheckError();
      }
      };

      /*Far contribution of M·F and B·dW
	Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·G_k·FFTf·S·F+ √σ·St·FFTi·√G_k·dWw =
	= σ·St·FFTi( G_k·FFTf·S·F + 1/√σ·√G_k·dWw)
      */
      void FarField::computeHydrodynamicDisplacements(real4* pos, real4* forces, real3 *MF, int numberParticles,
						      real temperature, real prefactor, cudaStream_t st){
	System::log<System::DEBUG1>("[BDHI::PSE] Computing MF wave space....");
	System::log<System::DEBUG2>("[BDHI::PSE] Setting vels to zero...");
	// G_k·FFT(S·F)
	//The computation is skipped if the forces are not provided (nullptr)
	auto gridVelsFourier = deterministicPart(pos, forces, numberParticles, st);
	//1/√σ·√G_k·dWw
	//The computation is skipped if temperature is zero
	addBrownianNoise(gridVelsFourier, temperature, prefactor, st);
	//FFTi
	auto gridVels = inverseTransformVelocity(gridVelsFourier, st);
	//St
	interpolateVelocity(gridVels, pos, MF, numberParticles, st);
	System::log<System::DEBUG2>("[BDHI::PSE] MF wave space Done");
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
	System::log<System::DEBUG>("[BDHI::PSE] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
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
	System::log<System::DEBUG>("[BDHI::PSE] Necessary work space for cuFFT: %s", printUtils::prettySize(cufftWorkSize).c_str());
	size_t free_mem, total_mem;
	CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
	if(free_mem<cufftWorkSize){
	  System::log<System::EXCEPTION>("[BDHI::PSE] Not enough memory in device to allocate cuFFT free %s, needed: %s!!, try lowering the splitting parameter!",
				      printUtils::prettySize(free_mem).c_str(),
				      printUtils::prettySize(cufftWorkSize).c_str());
	  throw std::runtime_error("Not enough memory for cuFFT");
	}
	this->cufftWorkAreaSize = cufftWorkSize+1;
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
	System::log<System::MESSAGE>("[BDHI::PSE] eta: %g", eta);
	kernel = std::make_shared<Kernel>(P, sqrt(eta)/(2.0*psi));
      }

      void FarField::initializeGrid(real tolerance){
	real kcut = 2*psi*sqrt(-log(tolerance));
	System::log<System::MESSAGE>("[BDHI::PSE] Far range wave number cut off: %f", kcut);
	const double hgrid = 2*M_PI/kcut;
	int3 cellDim = make_int3(2*box.boxSize/hgrid)+1;
	cellDim = nextFFTWiseSize3D(cellDim);
	this->grid = Grid(box, cellDim);
      }
    }
  }
}


#endif

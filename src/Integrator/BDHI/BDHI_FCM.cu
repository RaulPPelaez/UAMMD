/*Raul P. Pelaez 2018-2020. Force Coupling Method BDHI Module.
  See BDHI_FCM.cuh for information.
*/
#include"BDHI_FCM.cuh"
#include "misc/IBM.cuh"
#include"third_party/saruprng.cuh"
#include "utils/Grid.cuh"
#include "utils/NVTXTools.h"
#include "utils/cufftDebug.h"
#include"utils/debugTools.h"
#include"third_party/type_names.h"
namespace uammd{
  namespace BDHI{

    FCM::FCM(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys,
	     Parameters par):
      Integrator(pd,pg,sys,"BDHI:FCM"),
      pd(pd), pg(pg), sys(sys),
      dt(par.dt),
      par(par),
      temperature(par.temperature),
      viscosity(par.viscosity),
      hydrodynamicRadius(par.hydrodynamicRadius),
      steps(par.steps),
      box(par.box){
      sys->log<System::MESSAGE>("[BDHI::FCM] Initialized");
      if(box.boxSize.x == real(0.0) && box.boxSize.y == real(0.0) && box.boxSize.z == real(0.0)){
	sys->log<System::CRITICAL>("[BDHI::FCM] Box of size zero detected, cannot work without a box! (make sure a box parameter was passed)");
      }
      cudaStreamCreate(&st);
      seed = sys->rng().next32();
      initializeGrid(par);
      initializeKernel(par);
      initializeKernelTorque(par);
      printMessages(par);
      //int ncells = grid.cellDim.x*grid.cellDim.y*grid.cellDim.z;
      //gridVelsFourier.resize(3*ncells, cufftComplex());
      //gridVels.resize(ncells, real3());
      initCuFFT();
      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
    }

    void FCM::initializeGrid(Parameters par){
      int3 cellDim;
      real h;
      if(par.cells.x<=0){
       	if(par.hydrodynamicRadius<=0)
       	  sys->log<System::CRITICAL>("[BDHI::FCM] I need an hydrodynamic radius if cell dimensions are not provided!");
        h = Kernel::adviseGridSize(par.hydrodynamicRadius, par.tolerance);
	cellDim = nextFFTWiseSize3D(make_int3(par.box.boxSize/h));
      }
      else{
	cellDim = par.cells;
      }
      h = box.boxSize.x/cellDim.x;
      this->grid = Grid(box, cellDim);
    }

    void FCM::initializeKernel(Parameters par){
      real h = std::min({grid.cellSize.x, grid.cellSize.y, grid.cellSize.z});
      kernel = std::make_shared<Kernel>(h, par.tolerance);
      this->hydrodynamicRadius = kernel->fixHydrodynamicRadius(h, grid.cellSize.x);
    }

    void FCM::initializeKernelTorque(Parameters par){
      real h = pow(M_PI/6.0,real(1.0)/real(3.0))*std::min({grid.cellSize.x, grid.cellSize.y, grid.cellSize.z});
      kernelTorque = std::make_shared<Kernel>(h, par.tolerance);
    }

    void FCM::printMessages(Parameters par){
      auto rh = this->getHydrodynamicRadius();
      auto M0 = this->getSelfMobility();
      sys->log<System::MESSAGE>("[BDHI::FCM] Using kernel: %s", type_name<Kernel>().c_str());
      sys->log<System::MESSAGE>("[BDHI::FCM] Closest possible hydrodynamic radius: %g (%g requested)", rh, par.hydrodynamicRadius);
      sys->log<System::MESSAGE>("[BDHI::FCM] Self mobility: %g", (double)M0);
      if(box.boxSize.x != box.boxSize.y || box.boxSize.y != box.boxSize.z || box.boxSize.x != box.boxSize.z){
	sys->log<System::WARNING>("[BDHI::FCM] Self mobility will be different for non cubic boxes!");
      }
      sys->log<System::MESSAGE>("[BDHI::FCM] Box Size: %g %g %g", grid.box.boxSize.x, grid.box.boxSize.y, grid.box.boxSize.z);
      sys->log<System::MESSAGE>("[BDHI::FCM] Grid dimensions: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
      sys->log<System::MESSAGE>("[BDHI::FCM] Interpolation kernel support: %g rh max distance, %d cells total",
				kernel->support*0.5*grid.cellSize.x/rh, kernel->support);
      sys->log<System::MESSAGE>("[BDHI::FCM] h: %g %g %g", grid.cellSize.x, grid.cellSize.y, grid.cellSize.z);
      sys->log<System::MESSAGE>("[BDHI::FCM] Requested kernel tolerance: %g", par.tolerance);
      if(kernel->support >= grid.cellDim.x or
	 kernel->support >= grid.cellDim.y or
	 kernel->support >= grid.cellDim.z){
	sys->log<System::ERROR>("[BDHI::FCM] Kernel support is too big, try lowering the tolerance or increasing the box size!.");
      }
    }

    void FCM::initCuFFT(){
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
      sys->log<System::DEBUG>("[BDHI::FCM] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
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
      sys->log<System::DEBUG>("[BDHI::FCM] Necessary work space for cuFFT: %s, available: %s, total: %s",
			      printUtils::prettySize(cufftWorkSize).c_str(),
			      printUtils::prettySize(free_mem).c_str(),
			      printUtils::prettySize(total_mem).c_str());
      if(free_mem<cufftWorkSize){
	sys->log<System::EXCEPTION>("[BDHI::FCM] Not enough memory in device to allocate cuFFT free %s, needed: %s!!",
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(cufftWorkSize).c_str());
	throw std::runtime_error("Not enough memory for cuFFT");
      }
      cufftWorkArea.resize(cufftWorkSize);
      auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));
    }

    FCM::~FCM(){
      cudaDeviceSynchronize();
      cufftDestroy(cufft_plan_inverse);
      cufftDestroy(cufft_plan_forward);
    }

    /*Compute M·F and B·dW in Fourier space
      σ = dx*dy*dz; h^3 in [1]
      Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
      = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
    */


    namespace EulerMaruyama_ns{
      /*
	dR = dt(KR+MF) + sqrt(2*T*dt)·BdW +T·divM·dt -> divergence is commented out for the moment
      */
      /*With all the terms computed, update the positions*/
      /*T=0 case is templated*/
      template<class IndexIterator>
      __global__ void integrateGPUD(real4* pos,
				    real4* dir,
				    IndexIterator indexIterator,
				    const real3* linearV,
				    const real3* angularV,
				    int N,
				    real dt){
	uint id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>=N) return;
	int i = indexIterator[id];

	/*Position and color*/
	real4 pc = pos[i];
	real3 p = make_real3(pc);
	real c = pc.w;

	/*Update the position*/
	p += linearV[id]*dt;

	/*Write to global memory*/
	pos[i] = make_real4(p,c);
	
	/*Update the orientation*/
	if(dir){ 
	  Quat dirc = dir[i];
	  real3 dphi = angularV[id]*dt;
	  dirc = rotVec2Quaternion(dphi)*dirc;
	  dir[i] = dirc.to_real4();
	}
      }
    }

    
    template<typename vtype>
    void FCM::Mdot(real3 *Mv, vtype *v, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::FCM] Mdot....");
      thrust::fill(thrust::cuda::par.on(st), Mv, Mv + pg->getNumberParticles(), real3());
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      
      int numberParticles = pg->getNumberParticles();
      int3 n = grid.cellDim;
      cached_vector<cufftComplex3> gridVelsFourier(n.x*n.y*n.z);
      {
      auto gridVels =  spreadForces(pos.begin(), v, st);
      gridVelsFourier = forwardTransform(gridVels,st);
      convolveFourier(gridVelsFourier, st);
      addBrownianNoise(gridVelsFourier,st);
      }
      auto gridVels = inverseTransform(gridVelsFourier,st);      
      //interpolateVelocity(pos.begin(), Mv, gridVels,st);
      /*
      auto linearVelocities = interpolateVelocity(pos.begin(), gridVels, st);
      auto d_linearVelocities = thrust::raw_pointer_cast(linearVelocities.data());
      thrust::copy(d_linearVelocities, d_linearVelocities+numberParticles,Mv);
      */      
      sys->log<System::DEBUG2>("[BDHI::FCM] MF wave space Done");
    }    

    std::pair<cached_vector<real3>, cached_vector<real3>>
    FCM::Mdot(real4* pos, real4* force, real4* torque, cudaStream_t st){
      
      int numberParticles = pg->getNumberParticles();
      int3 n = grid.cellDim;
      cached_vector<cufftComplex3> gridVelsFourier(n.x*n.y*n.z);
      {
      auto gridVels = spreadForces(pos,force, st);
      gridVelsFourier = forwardTransform(gridVels,st);
      if (torque){
	addSpreadTorquesFourier(pos, torque, gridVelsFourier, st);
      }
      convolveFourier(gridVelsFourier, st);
      addBrownianNoise(gridVelsFourier, st);
      }
      auto gridVels = inverseTransform(gridVelsFourier,st);      
      
      cached_vector<real3> angularVelocities(numberParticles);
      cached_vector<real3> linearVelocities = interpolateVelocity(pos, gridVels, st);      
      
      if (torque){
	auto gridAngVelFourier = computeGridAngularVelocityFourier(gridVelsFourier, st);
	angularVelocities = interpolateAngularVelocity(pos, gridAngVelFourier,st);
      }
      sys->log<System::DEBUG2>("[BDHI::FCM] MDot wave space Done");
      CudaCheckError();
      
      return {linearVelocities, angularVelocities};
    }
    
    namespace FCM_ns{

      using cufftComplex3 = FCM::cufftComplex3;

      __device__ int3 indexToWaveNumber(int i, int3 nk){
	int ikx = i%(nk.x/2+1);
	int iky = (i/(nk.x/2+1))%nk.y;
	int ikz = i/((nk.x/2+1)*nk.y);
	ikx -= nk.x*(ikx >= (nk.x/2+1));
	iky -= nk.y*(iky >= (nk.y/2+1));
	ikz -= nk.z*(ikz >= (nk.z/2+1));
	return make_int3(ikx, iky, ikz);
      }

      __device__ real3 waveNumberToWaveVector(int3 ik, real3 L){
	return (real(2.0)*real(M_PI)/L)*make_real3(ik.x, ik.y, ik.z);
      }

      /*Apply the projection operator to a wave number with a certain complex factor.
	res = (I-\hat{k}^\hat{k})·factor*/
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
	
      __global__ void addTorqueCurl(cufftComplex3 *gridTorquesFourier, cufftComplex3* gridVelsFourier, Grid grid){
	  
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	
	const int3 nk = grid.cellDim; 
	if(id >= (nk.z*nk.y*(nk.x/2+1))) return;
	  
	const int3 ik = indexToWaveNumber(id, nk);
	const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize);
	
	const real half = real(0.5);
	  
	const bool isUnpairedX = ik.x == (nk.x - ik.x);
	const bool isUnpairedY = ik.y == (nk.y - ik.y);	
	const bool isUnpairedZ = ik.y == (nk.z - ik.z);	
	real Dx = isUnpairedX?0:k.x;
	real Dy = isUnpairedY?0:k.y;
	real Dz = isUnpairedZ?0:k.z;
	  
	auto gridi = gridTorquesFourier[id];
	cufftComplex3 gridVeli;
	  
	gridVeli.x = {half*(-Dy*gridi.z.y+Dz*gridi.y.y),
	    half*(Dy*gridi.z.x-Dz*gridi.y.x)};
	gridVeli.y = {half*(-Dz*gridi.x.y+Dy*gridi.z.y),
	    half*(-Dz*gridi.x.x-Dx*gridi.z.x)};
	gridVeli.z = {half*(-Dx*gridi.y.y + Dy*gridi.x.y),
	    half*(Dx*gridi.y.x - Dy*gridi.x.x)};
	gridVelsFourier[id] += gridVeli; 
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

      	/*Compute gaussian complex noise dW, std = prefactor -> ||z||^2 = <x^2>/sqrt(2)+<y^2>/sqrt(2) = prefactor*/
	/*A complex random number for each direction*/
	__device__ cufftComplex3 generateNoise(real prefactor, uint id, uint seed1, uint seed2){	  //Uncomment to use uniform numbers instead of gaussian
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

      struct ToReal3{
	template<class vtype>
	inline __device__ real3 operator()(vtype q){return make_real3(q);}
      };

    }
    
    cached_vector<real3> FCM::spreadForces(real4* pos, real4* force, cudaStream_t st){
      /*Spread force on particles to grid positions -> S·F*/
      sys->log<System::DEBUG2>("[BDHI::FCM] Particles to grid");
      int numberParticles = pg->getNumberParticles();
      auto force_r3 = thrust::make_transform_iterator(force, FCM_ns::ToReal3());
      int3 n = grid.cellDim;
      cached_vector<real3> gridVels(n.x*n.y*n.z); 
      thrust::fill(thrust::cuda::par.on(st), gridVels.begin(), gridVels.end(), real3());
      auto d_gridVels = (real3*)thrust::raw_pointer_cast(gridVels.data());
      IBM<Kernel> ibm(sys, kernel, grid);
      ibm.spread(pos, force_r3, d_gridVels, numberParticles, st);
      CudaCheckError();
      return gridVels;
    }
  
    cached_vector<FCM::cufftComplex3> FCM::forwardTransform(cached_vector<real3>& gridReal, cudaStream_t st){
      int3 n = grid.cellDim;
      cached_vector<cufftComplex3> gridFourier(n.x*n.y*n.z);
      thrust::fill(thrust::cuda::par.on(st), gridFourier.begin(), gridFourier.end(), cufftComplex3());
      auto d_gridFourier = (cufftComplex*) thrust::raw_pointer_cast(gridFourier.data());
      auto d_gridReal = (real*) thrust::raw_pointer_cast(gridReal.data());
      cufftSetStream(cufft_plan_forward, st);
      /*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
      CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, d_gridReal, d_gridFourier));
      return gridFourier;
    }
    
    void FCM::addSpreadTorquesFourier(real4* pos, real4* torque, cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st){
      /*Spread force on particles to grid positions -> S·F*/
      sys->log<System::DEBUG2>("[BDHI::FCM] Spreading torques");

      int numberParticles = pg->getNumberParticles();
      int3 n = grid.cellDim;
      
      auto torque_r3 = thrust::make_transform_iterator(torque, FCM_ns::ToReal3());
      

      cached_vector<real3> gridTorques(n.x*n.y*n.z);
      auto d_gridTorques3 = thrust::raw_pointer_cast(gridTorques.data());
      // Si uso cached_vector no corre bien, "illegal memory access"
            
      IBM<Kernel> ibm(sys, kernelTorque, grid);
      ibm.spread(pos, torque_r3, d_gridTorques3, numberParticles, st);
      auto gridTorquesFourier = forwardTransform(gridTorques, st);
      
      
      int BLOCKSIZE = 128;
      int numberCells = n.z*n.y*(n.x/2+1);
      uint Nthreads = BLOCKSIZE<numberCells?BLOCKSIZE:numberCells;
      uint Nblocks = numberCells/Nthreads +  ((numberCells%Nthreads!=0)?1:0);
      
      auto d_gridVelsFourier = (cufftComplex3*) thrust::raw_pointer_cast(gridVelsFourier.data());
      auto d_gridTorquesFourier3 = (cufftComplex3*) thrust::raw_pointer_cast(gridTorquesFourier.data());
      FCM_ns::addTorqueCurl<<<Nblocks, Nthreads,0,st>>>(d_gridTorquesFourier3,d_gridVelsFourier, grid);
      CudaCheckError();
    }
  
    void FCM::convolveFourier(cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st){
      sys->log<System::DEBUG2>("[BDHI::FCM] Wave space velocity scaling");
      /*Scale the wave space grid forces, transforming in velocities -> B·FFT·S·F*/
      auto d_gridVelsFourier = (cufftComplex3*) thrust::raw_pointer_cast(gridVelsFourier.data());
      const int3 n = grid.cellDim;
      int Nthreads = 128;
      int Nblocks = (n.z*n.y*(n.x/2+1))/Nthreads +1;
      FCM_ns::forceFourier2Vel<<<Nblocks, Nthreads, 0, st>>> (d_gridVelsFourier, d_gridVelsFourier, viscosity, grid);
    
      CudaCheckError();
    }

    void FCM::addBrownianNoise(cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st){
      //The sqrt(2*T/dt) factor needs to be here because far noise is summed to the M·F term.
      /*Add the stochastic noise to the fourier velocities if T>0 -> 1/√σ·√B·dWw */
      if(temperature > real(0.0)){
	auto d_gridVelsFourier =  (cufftComplex3*)thrust::raw_pointer_cast(gridVelsFourier.data());
	uint seed2 = sys->rng().next32();
	sys->log<System::DEBUG2>("[BDHI::FCM] Wave space brownian noise");
	const int3 n = grid.cellDim;
	const real dV = grid.getCellVolume();
	const real fourierNormalization = 1.0/(double(n.x)*n.y*n.z);
	real prefactor = sqrt(fourierNormalization*2*temperature/(dt*dV));
	int Nthreads = 128;
	int Nblocks = (n.z*n.y*(n.x/2+1))/Nthreads +1;
	//In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw
	FCM_ns::fourierBrownianNoise<<<Nblocks, Nthreads, 0, st>>>(d_gridVelsFourier, grid,
								   prefactor, // 1/√σ· sqrt(2*T/dt),
								   viscosity,
								   seed,
								   seed2);
	CudaCheckError();
      }
    }
    
    cached_vector<real3> FCM::inverseTransform(cached_vector<cufftComplex3>& gridFourier, cudaStream_t st){
      int3 n = grid.cellDim;
      cached_vector<real3> gridReal(n.x*n.y*n.z);
      thrust::fill(thrust::cuda::par.on(st), gridReal.begin(), gridReal.end(), real3());
      auto d_gridFourier = (cufftComplex*) thrust::raw_pointer_cast(gridFourier.data());
       auto d_gridReal = (real*) thrust::raw_pointer_cast(gridReal.data());
      
      cufftSetStream(cufft_plan_inverse, st);
      /*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
      CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, d_gridFourier, d_gridReal));
      return gridReal;
    }
    
    
    void FCM::interpolateVelocity(real4* pos, real3* linearVelocities, cached_vector<real3>& gridVels, cudaStream_t st){
      sys->log<System::DEBUG2>("[BDHI::FCM] Grid to particles");
      /*Interpolate the real space velocities back to the particle positions ->
	Output: Mv = Mw·F + sqrt(2*T/dt)·√Mw·dWw = σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
      int numberParticles = pg->getNumberParticles();
      real3* d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      IBM<Kernel> ibm(sys, kernel, grid);
      ibm.gather(pos, linearVelocities, d_gridVels, numberParticles, st);
      CudaCheckError();
    }

    
    cached_vector<real3> FCM::interpolateVelocity(real4* pos, cached_vector<real3>& gridVels, cudaStream_t st){
      sys->log<System::DEBUG2>("[BDHI::FCM] Grid to particles");
      /*Interpolate the real space velocities back to the particle positions ->
	Output: Mv = Mw·F + sqrt(2*T/dt)·√Mw·dWw = σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
      int numberParticles = pg->getNumberParticles();
      real3* d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      IBM<Kernel> ibm(sys, kernel, grid);
      cached_vector<real3> linearVelocities(numberParticles);
      thrust::fill(thrust::cuda::par.on(st), linearVelocities.begin(), linearVelocities.end(), real3());
      
      auto d_linearVelocities = thrust::raw_pointer_cast(linearVelocities.data());
      ibm.gather(pos, d_linearVelocities, d_gridVels, numberParticles, st);
      CudaCheckError();
      return linearVelocities;
    }
    
    namespace FCM_ns{

      //Compute the curl of the velocity, V, in chebyshev space. This is equal to the angular velocity
      // 0.5\nabla \times V = 0.5 (i*k_x i*k_y i*k_z)\times (V_x V_y V_z) =
      // = 0.5( i*k_y*V_z - i*k_z(V_y), i*k_z(V_x) - i*k_x*V_z, i*k_x*V_y - i*k_y*V_x)
      //Overwrite the output vector with the angular velocities in Cheb space
      //The input velocity vector is overwritten

      using cufftComplex3 = FCM::cufftComplex3;

      __global__ void computeVelocityCurlFourier(cufftComplex3 *gridVelsFourier, cufftComplex3* gridAngVelsFourier, Grid grid){
	
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	const int3 nk = grid.cellDim;
	if(id >= nk.z*nk.y*(nk.x/2+1)){
	return;
	}
	
	const int3 cell = make_int3(id%(nk.x/2+1), (id/(nk.x/2+1))%nk.y, id/((nk.x/2+1)*nk.y));

	
	const int3 ik = indexToWaveNumber(id, nk);
	const real3 k = waveNumberToWaveVector(ik, grid.box.boxSize);
	const real half = real(0.5);
	
	const bool isUnpairedX = ik.x == (nk.x - ik.x);
	const bool isUnpairedY = ik.y == (nk.y - ik.y);	
	const bool isUnpairedZ = ik.y == (nk.z - ik.z);	
	real Dx = isUnpairedX?0:k.x;
	real Dy = isUnpairedY?0:k.y;
	real Dz = isUnpairedZ?0:k.z;

	cufftComplex3 gridAng;
	auto gridLinear = gridVelsFourier[id];
	
	gridAng.x = {half*(-Dy*gridLinear.z.y+Dz*gridLinear.y.y),
	  half*(Dy*gridLinear.z.x-Dz*gridLinear.y.x)};
	gridAng.y = {half*(-Dz*gridLinear.x.y+Dy*gridLinear.z.y),
	  half*(-Dz*gridLinear.x.x-Dx*gridLinear.z.x)};
	gridAng.z = {half*(-Dx*gridLinear.y.y + Dy*gridLinear.x.y),
	  half*(Dx*gridLinear.y.x - Dy*gridLinear.x.x)};
      
	gridAngVelsFourier[id] = gridAng;
      }
    }
    
    cached_vector<FCM::cufftComplex3> FCM::computeGridAngularVelocityFourier(cached_vector<cufftComplex3>& gridVelsFourier, cudaStream_t st){

      const int3 n = grid.cellDim;
      const int blockSize = 128;
      const int numberSystems = n.z*n.y*(n.x/2+1);
      const int numberBlocks = numberSystems/blockSize+1;
      
      auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());
      cached_vector<cufftComplex3> gridAngVelsFourier(gridVelsFourier.size());
      auto d_gridAngVelsFourier =  thrust::raw_pointer_cast(gridAngVelsFourier.data());
      FCM_ns::computeVelocityCurlFourier<<<numberBlocks, blockSize, 0, st>>>(d_gridVelsFourier, d_gridAngVelsFourier,
									     grid);
      CudaCheckError();
      return gridAngVelsFourier;
    }
    
    cached_vector<real3> FCM::interpolateAngularVelocity(real4* pos, cached_vector<cufftComplex3>& gridAngVelsFourier, cudaStream_t st){

 
      int numberParticles = pg->getNumberParticles();
          
      auto gridAngVels = inverseTransform(gridAngVelsFourier, st);
      IBM<Kernel> ibm(sys, kernelTorque, grid);
      cached_vector<real3> angularVelocities(numberParticles);
      thrust::fill(thrust::cuda::par.on(st), angularVelocities.begin(), angularVelocities.end(), real3());
      auto d_angularVelocities = thrust::raw_pointer_cast(angularVelocities.data());
      auto d_gridAngVels = thrust::raw_pointer_cast(gridAngVels.data());
      ibm.gather(pos, d_angularVelocities, d_gridAngVels, numberParticles, st);
      CudaCheckError();
      return angularVelocities;
    }

    
    void FCM::computeMF(real3* MF, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::FCM] Computing MF....");
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      Mdot<real4>(MF, force.begin(), st);
    }
    
    void FCM::computeBdW(real3* BdW, cudaStream_t st){
      //This part is included in Fourier space when computing MF
    }

    void FCM::updateInteractors(){

      for(auto forceComp: interactors) forceComp->updateSimulationTime(steps*par.dt);
      
      if(steps==1){
	for(auto forceComp: interactors){
	  forceComp->updateTimeStep(par.dt);
	  forceComp->updateTemperature(par.temperature);
	  forceComp->updateBox(par.box);
	}
      }
    }

    void FCM::resetForces(){
      int numberParticles = pg->getNumberParticles();
      auto force = pd->getForce(access::location::gpu, access::mode::write);
      auto forceGroup = pg->getPropertyIterator(force);
      thrust::fill(thrust::cuda::par.on(st), forceGroup, forceGroup + numberParticles, real4());
      CudaCheckError();
    }

    void FCM::resetTorques(){
      int numberParticles = pg->getNumberParticles();
      auto torque = pd->getTorque(access::location::gpu, access::mode::write);
      auto torqueGroup = pg->getPropertyIterator(torque);
      thrust::fill(thrust::cuda::par.on(st), torqueGroup, torqueGroup + numberParticles, real4());
      CudaCheckError();
    }

    void FCM::computeCurrentForces(){
      resetForces();
      if (pd->isDirAllocated()) resetTorques();
      for(auto forceComp: interactors) forceComp->sumForce(st);
      CudaCheckError();
    }
    
    void FCM::forwardTime(){
      steps++;
      sys->log<System::DEBUG1>("[BDHI::FCM] Performing integration step %d", steps);
      updateInteractors();
      computeCurrentForces(); //Compute forces and torques
      
      int numberParticles = pg->getNumberParticles();
      cached_vector<real3> linearVelocities(numberParticles);
      cached_vector<real3> angularVelocities(numberParticles);

      {
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      auto torque = pd -> getTorqueIfAllocated(access::location::gpu, access::mode::read);
      auto dir = pd->getDirIfAllocated(access::location::gpu, access::mode::readwrite);
      
      std::tie(linearVelocities,angularVelocities) = Mdot(pos.raw(),
							  force.raw(),
							  torque.raw(),
							  st);
      CudaCheckError();
      }
  
  
      auto indexIter = pg->getIndexIterator(access::location::gpu);
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      auto dir = pd->getDirIfAllocated(access::location::gpu, access::mode::readwrite);
      
      real3* d_linearV = thrust::raw_pointer_cast(linearVelocities.data());

      real3* d_angularV = nullptr;
      if(dir.raw()) d_angularV = thrust::raw_pointer_cast(angularVelocities.data());
 
      int BLOCKSIZE = 128; /*threads per block*/
      int nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int nblocks = numberParticles/nthreads +  ((numberParticles%nthreads!=0)?1:0);

      
      
      EulerMaruyama_ns::integrateGPUD<<<nblocks, nthreads, 0, st>>>(pos.raw(),
								    dir.raw(),
								    indexIter,
								    d_linearV,
								    d_angularV,
								    numberParticles,
								    par.dt);
      CudaCheckError();
    }
  }    
}


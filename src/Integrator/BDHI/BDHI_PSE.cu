/*Raul P. Pelaez 2017. Positively Split Edwald Rotne-Prager-Yamakawa Brownian Dynamics with Hydrodynamic interactions.


  As this is a BDHI module. BDHI_PSE computes the terms M·F and B·dW in the differential equation:
            dR = K·R·dt + M·F·dt + sqrt(2Tdt)· B·dW

  The mobility, M, is computed according to the Rotne-Prager-Yamakawa (RPY) tensor.

  The computation uses periodic boundary conditions (PBC)
  and partitions the RPY tensor in two, positively defined contributions [1], so that:
      M = Mr + Mw
       Mr - A real space short range contribution.
       Mw - A wave space long range contribution.

  Such as:
     M·F = Mr·F + Mw·F
     B·dW = sqrt(Mr)·dWr + sqrt(Mw)·dWw
####################      Short Range     #########################


  Mr·F: The short range contribution of M·F is computed using a neighbour list (this is like a sparse matrix-vector product in which each element is computed on the fly), see PSE_ns::RPYNearTransverser.
        The RPY near part function (see Apendix 1 in [1]) is precomputed and stored in texture memory,
	see PSE_ns::RPYPSE_nearTextures.

  sqrt(Mr)·dW: The near part stochastic contribution is computed using the Lanczos algorithm (see misc/LanczosAlgorithm.cuh), the function that computes M·v is provided via a functor called PSE_ns::Dotctor, the logic of M·v itself is the same as in M·F (see PSE_ns::RPYNearTransverser) and is computed with the same neighbour list.

###################        Far range     ###########################



  Mw·F:  Mw·F = σ·St·FFTi·B·FFTf·S · F. The long range wave space part.
         -σ: The volume of a grid cell
	 -S: An operator that spreads each element of a vector to a regular grid using a gaussian kernel.
	 -FFT: Fast fourier transform operator.
	 -B: A fourier scaling factor in wave space to transform forces to velocities, see eq.9 in [1].

        Related functions:
	FFT: cufftExecR2C (forward), cufftC2R(inverse)
	S: PSE_ns::particles2Grid (S), PSE_ns::grid2Particles (σ·St)
	B: PSE_ns::fillFourierScalingFactor, PSE_ns::forceFourier2vel

  sqrt(Mw)·dWw: The far range stochastic contribution is computed in fourier space along M·F as:
               Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
                            = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	        Only one St·FFTi is needed, the stochastic term is added as a velocity in fourier space.                dWw is a gaussian random vector of complex numbers, special care must be taken to ensure the correct conjugacy properties needed for the FFT. See PSE_ns::fourierBrownianNoise

Therefore, in the case of Mdot_far, for computing M·F, Bw·dWw is also summed.

computeBdW computes only the real space stochastic contribution.

Notes about the FFT format:
cufftR2C and C2R are used with padding storage mode. That means that while only the first half of the innermost dimension (x) is computed and stored the storage has size Nx*Ny*Nz, which implies that the data is padded Nx/2 unused elements between "y" values. This simplifies the access pattern at the cost of wasting storage...

Currently the fourier data has the following format:
Each 3D fourier node / real space cell has three components x,y,z.
  The array V storing the nodes is structured as follows:
     the component "a" of the node x,y,z is located at V[Nx*Ny*z+Nx*y+x].a.
When storing real space quantities V[...].x/y/z are real numbers, whereas fourier quantities have the same access patter except being cufftComplex numbers.
In other words, V is interpreted as an array of real3 when in real space and cufftComplex3 when in fourier space.


Notes:
Storing F and G functions in r^2 scale (a.i. table(r^2) = F(sqrt(r^2))) creates artifacts due to the linear interpolation of a cuadratic scale, so it is best to just store table(sqrt(r^2)) = F(r). The cost of the sqrt seems neglegible and gives a really good change in accuracy.

References:

[1]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations
           -  https://arxiv.org/pdf/1611.09322.pdf
[2]  Spectral accuracy in fast Ewald-based methods for particle simulations
           -  http://www.sciencedirect.com/science/article/pii/S0021999111005092

TODO:
70- Use native cufft memory layout
10- Abstract NUFFT logic to a class for later reuse.

Special thanks to Marc Melendez and Florencio Balboa.
*/
#include"BDHI_PSE.cuh"
#include"utils/GPUUtils.cuh"
#include"misc/TabulatedFunction.cuh"
#include"RPY_PSE.cuh"
#include"third_party/saruprng.cuh"
#include<vector>
#include<algorithm>
#include<fstream>
#include"utils/debugTools.h"
#include"utils/cufftDebug.h"
#include"utils/curandDebug.h"

namespace uammd{
  namespace BDHI{

    namespace PSE_ns{

      //Looks for the closest (equal or greater) number of nodes of the form 2^a*3^b*5^c
      int3 nextFFTWiseSize3D(int3 size){

	int* cdim = &size.x;

	int max_dim = std::max({size.x, size.y, size.z});

	int n= 5;
	std::vector<int> tmp(n*n*n*n*n, 0);
	do{
	  tmp.resize(n*n*n*n*n, 0);
	  fori(0,n)forj(0,n)for(int k=0; k<n;k++)for(int k7=0; k7<n; k7++)for(int k11=0; k11<n; k11++){
		int id = i+n*j+n*n*k+n*n*n*k7+n*n*n*n*k11;
		tmp[id] = 0;
		//Current fft wise size
		int number = pow(2,i)*pow(3,j)*pow(5,k)*pow(7, k7)*pow(11, k11);
		//The fastest FFTs always have at least a factor of 2
		if(i==0) continue;
		//I have seen empiracally that factor 11 and 7 only works well with at least a factor 2 involved
		if((k11>0 && (i==0))) continue;
		tmp[id] = number;
	      }
	  n++;
	  /*Sort this array in ascending order*/
	  std::sort(tmp.begin(), tmp.end());
	}while(tmp.back()<max_dim); /*if n=5 is not enough, include more*/

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


      /*A convenient struct to pack 3 complex numbers, that is 6 real numbers*/
      struct cufftComplex3{
	cufftComplex x,y,z;
      };

      inline __device__ __host__ cufftComplex3 operator+(const cufftComplex3 &a, const cufftComplex3 &b){
	return {a.x + b.x, a.y + b.y, a.z + b.z};
      }
      inline __device__ __host__ void operator+=(cufftComplex3 &a, const cufftComplex3 &b){
	a.x += b.x; a.y += b.y; a.z += b.z;
      }

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


      /* Precomputes the fourier scaling factor B (see eq. 9 and 20.5 in [1]),
	 Bfactor = B(||k||^2, xi, tau) = 1/(vis·Vol) · sinc(k·rh)^2/k^2·Hashimoto(k,xi,tau)
      */
      __global__ void fillFourierScalingFactor(real3 * __restrict__ Bfactor, /*Global memory results*/
					       Grid grid,  /*Grid information*/
					       double rh, /*Hydrodynamic radius*/
					       double vis, /*Viscosity*/
					       double psi, /*RPY splitting parameter*/
					       real3 eta /*Gaussian kernel splitting parameter*/
					       ){
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

	/*K=0 doesnt contribute*/
	if(icell == 0){
	  Bfactor[0] = make_real3(0.0);
	  return;
	}
	/*The factors are computed in double precision for improved accuracy*/

	double3 K = cellToWaveNumber(cell, grid.cellDim, make_double3(grid.box.boxSize));

	/*Compute the scaling factor for this node*/
	double k2 = dot(K,K);
	double kmod = sqrt(k2);
	double invk2 = double(1.0)/k2;

	double sink = sin(kmod*rh);

	double k2_invpsi2_4 = k2/(4.0*psi*psi);

	/*The Hashimoto splitting function,
	  psi is the splitting between near and far contributions,
	  eta is the splitting of the gaussian kernel used in the grid interpolation, see sec. 2 in [2]*/
	/*See eq. 11 in [1] and eq. 11 and 14 in [2]*/
	double3 tau = make_double3(-k2_invpsi2_4*(1.0-eta));
	double3 hashimoto = (1.0 + k2_invpsi2_4)*make_double3(exp(tau.x), exp(tau.y), exp(tau.z))/k2;

	/*eq. 20.5 in [1]*/
	double3 B = sink*sink*invk2*hashimoto/(vis*rh*rh);
	B /= double(grid.cellDim.x*grid.cellDim.y*grid.cellDim.z);
	/*Store theresult in global memory*/
	Bfactor[icell] = make_real3(B);
      }

    }
    /*Constructor*/
    PSE::PSE(shared_ptr<ParticleData> pd,
	     shared_ptr<ParticleGroup> pg,
	     shared_ptr<System> sys,
	     Parameters par):
      pd(pd), pg(pg), sys(sys),
      hydrodynamicRadius(par.hydrodynamicRadius),
      dt(par.dt),
      temperature(par.temperature),
      box(par.box), grid(box, int3()),
      psi(par.psi){

      if(box.boxSize.x == real(0.0) && box.boxSize.y == real(0.0) && box.boxSize.z == real(0.0)){
	sys->log<System::CRITICAL>("[BDHI::PSE] Box of size zero detected, cannot work without a box! (make sure a box parameter was passed)");
      }
      if(box.boxSize.x != box.boxSize.y || box.boxSize.y != box.boxSize.z || box.boxSize.x != box.boxSize.z){
	sys->log<System::WARNING>("[BDHI::PSE] Non cubic boxes are not really tested!");
      }


      //It appears that this tolerance is unnecesary for lanczos, but I am not sure so better leave it like this.
      this->lanczosTolerance = par.tolerance; //std::min(0.05f, sqrt(par.tolerance));
      this->lanczos = std::make_shared<LanczosAlgorithm>(sys, lanczosTolerance);

      seed = sys->rng().next();
      sys->log<System::MESSAGE>("[BDHI::PSE] Initialized");

      int numberParticles = pg->getNumberParticles();

      /* M = Mr + Mw */
      sys->log<System::MESSAGE>("[BDHI::PSE] Self mobility: %f", 1.0/(6*M_PI*par.viscosity*par.hydrodynamicRadius)*(1-2.837297*par.hydrodynamicRadius/box.boxSize.x));

      const double pi = M_PI;
      const double a = par.hydrodynamicRadius;

      /****Initialize near space part: Mr *******/
      const real er = par.tolerance; /*Short range error tolerance*/
      /*Near neighbour list cutoff distance, see sec II:C in [1]*/
      rcut = sqrt(-log(er))/psi;

      if(0.5*box.boxSize.x < rcut){
	sys->log<System::WARNING>("[BDHI::PSE] A real space cut off (%e) larger than half the box size (%e) can cause mobility artifacts!, try increasing the splitting parameter (%e)", rcut, 0.5*box.boxSize.x, psi);
	rcut = box.boxSize.x*0.5;
      }

      /*Initialize the neighbour list */
      this->cl = std::make_shared<CellList>(pd, pg, sys);

      /*Initialize the near RPY textures*/
      {
	RPYPSE_near rpy(par.hydrodynamicRadius, psi, (6*M_PI*a*par.viscosity), rcut);

	real textureTolerance = a*par.tolerance; //minimum distance described
	int nPointsTable = int(rcut/textureTolerance + 0.5);

	nPointsTable = std::max(4096, nPointsTable);
	sys->log<System::MESSAGE>("[BDHI::PSE] Number of real RPY texture points: %d", nPointsTable);
	tableDataRPY.resize(nPointsTable+1);
	RPY_near = std::make_shared<TabulatedFunction<real2>>(thrust::raw_pointer_cast(tableDataRPY.data()),
							      nPointsTable,
							      0.0, //minimum distance
							      rcut,//maximum distance
							      rpy //Function to tabulate
							      );
      }
      /****Initialize wave space part: Mw ******/
      const real ew = par.tolerance; /*Long range error tolerance*/
      /*Maximum wave number for the far calculation*/
      kcut = 2*psi*sqrt(-log(ew));

      /*Corresponding real space grid size*/
      const double hgrid = 2*pi/kcut;
      /*Create a grid with cellDim cells*/
      /*This object will contain useful information about the grid,
	mainly number of cells, cell size and usual parameters for
	using it in the gpu, as invCellSize*/
      int3 cellDim = make_int3(2*box.boxSize/hgrid)+1;

      if(cellDim.x<3)cellDim.x = 3;
      if(cellDim.y<3)cellDim.y = 3;
      if(cellDim.z<3)cellDim.z = 3;

      /*FFT likes a number of cells as cellDim.i = 2^n·3^l·5^m */
      cellDim = PSE_ns::nextFFTWiseSize3D(cellDim);

      /*Store grid parameters in a Mesh object*/
      grid = Grid(box, cellDim);

      /*Print information*/
      sys->log<System::MESSAGE>("[BDHI::PSE] Box Size: %f %f %f", box.boxSize.x, box.boxSize.y, box.boxSize.z);
      sys->log<System::MESSAGE>("[BDHI::PSE] Unitless splitting factor ξ·a: %f", psi*par.hydrodynamicRadius);
      sys->log<System::MESSAGE>("[BDHI::PSE] Close range distance cut off: %f", rcut);
      sys->log<System::MESSAGE>("[BDHI::PSE] Far range wave number cut off: %f", kcut);
      sys->log<System::MESSAGE>("[BDHI::PSE] Far range grid size: %d %d %d", cellDim.x, cellDim.y, cellDim.z);
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

      /*Grid spreading/interpolation parameters*/
      /*Gaussian spreading/interpolation kernel support points neighbour distance
	See eq. 19 and sec 4.1 in [2]*/

      //m = C·sqrt(pi·P), from sec 4.1 we choose C=0.976
      constexpr double C = 0.976;
      double m = 1;
      /*I have empirically found that 0.1*tolerance here gives the desired overal accuracy*/
      while(erfc(m/sqrt(2)) > 0.1*par.tolerance) m+= 0.01;

      //This is P in [2]
      int support;
      //Support must be odd
      while( (support = int(pow(m/C, 2)/M_PI+0.5)+1 ) % 2 == 0) m+=par.tolerance;

      //P is in each direction, nearest neighbours -> P=1
      this->P = make_int3(support/2);

      //If P is too large for the grid set it to the grid size-1 (or next even number)
      int minCellDim = std::min({cellDim.x, cellDim.y, cellDim.z});
      if(support>minCellDim){
	support = minCellDim;
	if(support%2==0) support--; //minCellDim will be 3 at least

	P = make_int3(support/2);
	m = C*sqrt(M_PI*support);

      }

      sys->log<System::MESSAGE>("[BDHI_PSE] Gaussian kernel support in each direction: %d", this->P.x);


      double3 pw = make_double3(support);/*Number of support points*/

      double3 h = make_double3(grid.cellSize); /*Cell size*/
      /*Number of standard deviations in the grid's Gaussian kernel support*/
      double3 gaussM = make_double3(m);

      /*Standard deviation of the Gaussian kernel*/
      double3 w   = pw*h/2.0;
      /*Gaussian splitting parameter*/
      this->eta = make_real3(pow(2.0*psi, 2)*w*w/(gaussM*gaussM));

      /*B in [1], this array stores, for each cell/fourier node,
	the scaling factor to go from forces to velocities in fourier space*/
      fourierFactor.resize(ncells);

      /*Launch a thread per cell/node*/
      dim3 NthreadsCells = 128;
      dim3 NblocksCells;
      NblocksCells.x= grid.cellDim.x/NthreadsCells.x +1;
      NblocksCells.y= grid.cellDim.y/NthreadsCells.y +1;
      NblocksCells.z= grid.cellDim.z/NthreadsCells.z +1;

      PSE_ns::fillFourierScalingFactor<<<NblocksCells, NthreadsCells, 0, stream2>>>
	(thrust::raw_pointer_cast(fourierFactor.data()), grid,
	 par.hydrodynamicRadius, par.viscosity, psi, eta);



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
				      CUFFT_R2C, 3,
				      &cufftWorkSizef));

      sys->log<System::DEBUG>("[BDHI::PSE] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
      /*Same as above, but with C2R for inverse FFT*/
      CufftSafeCall(cufftMakePlanMany(cufft_plan_inverse,
				      3, &cdtmp.x, /*Three dimensional FFT*/
				      &cdtmp.x,
				      /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
				      3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
				      &cdtmp.x,
				      3, 1,
				      /*Perform 3 inverse batched FFTs*/
				      CUFFT_C2R, 3,
				      &cufftWorkSizei));

      /*Allocate cuFFT work area*/
      size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei);

      sys->log<System::DEBUG>("[BDHI::PSE] Necessary work space for cuFFT: %s", printUtils::prettySize(cufftWorkSize).c_str());
      size_t free_mem, total_mem;
      CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));

      if(free_mem<cufftWorkSize){
	sys->log<System::CRITICAL>("[BDHI::PSE] Not enough memory in device to allocate cuFFT free %s, needed: %s!!, try lowering the splitting parameter!",
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(cufftWorkSize).c_str());
      }

      cufftWorkArea.resize(cufftWorkSize+1);
      auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());

      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea));

      //Init rng
      CurandSafeCall(curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT));
      //curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_MT19937);

      CurandSafeCall(curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next()));

      thrust::device_vector<real> noise(30000);
      auto noise_ptr = thrust::raw_pointer_cast(noise.data());
      //Warm cuRNG
      CurandSafeCall(curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0));
      CurandSafeCall(curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0));


      CudaSafeCall(cudaDeviceSynchronize());
      CudaCheckError();
    }



    PSE::~PSE(){
      CudaSafeCall(cudaDeviceSynchronize());
      CufftSafeCall(cufftDestroy(cufft_plan_inverse));
      CufftSafeCall(cufftDestroy(cufft_plan_forward));
      CurandSafeCall(curandDestroyGenerator(curng));
      CudaSafeCall(cudaStreamDestroy(stream));
      CudaSafeCall(cudaStreamDestroy(stream2));
    }



    /*I dont need to do anything at the begining of a step*/
    void PSE::setup_step(cudaStream_t st){}


    /*Compute M·v = Mr·v + Mw·v*/
    template<typename vtype>
    void PSE::Mdot(real3 *Mv, vtype *v, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::PSE] Mdot....");
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);

      /*Ensure the result array is set to zero*/
      fillWithGPU<<<Nblocks, Nthreads>>>(Mv, make_real3(0.0), numberParticles);

      /*Update the list if needed*/
      cl->updateNeighbourList(box, rcut, st);

      Mdot_near<vtype>(Mv, v, st);
      //Mdot_farThread = std::thread(&PSE::Mdot_far<vtype>, this, Mv, v, stream2);
      //Mdot_far<vtype>(Mv, v, stream2);
      Mdot_far<vtype>(Mv, v, st);

    }



    namespace PSE_ns{
      /*Compute the product M_nearv = M_near·v by transversing a neighbour list

	This operation can be seen as an sparse MatrixVector product.
	Mv_i = sum_j ( Mr_ij·vj )
      */
      /*Each thread handles one particle with the other N, including itself*/
      /*That is 3 full lines of M, or 3 elements of M·v per thread, being the x y z of ij with j=0:N-1*/
      /*In other words. M is made of NxN boxes of size 3x3,
	defining the x,y,z mobility between particle pairs,
	each thread handles a row of boxes and multiplies it by three elements of v*/
      /*vtype can be real3 or real4*/
      /*Very similar to the NBody transverser in Lanczos, but this time the RPY tensor
	is read from a texture due to its complex form and the transverser is handed to a neighbourList*/
      template<class vtype>
      struct RPYNearTransverser{
	typedef real3 computeType; /*Each particle outputs a real3*/
	typedef real3 infoType;    /*And needs a real3 with information*/

	/*Constructor*/
	RPYNearTransverser(vtype* v,
			   real3 *Mv,
			   /*RPY_near(r) = F(r)·(I-r^r) + G(r)·r^r*/
			   TabulatedFunction<real2> FandG,
			   real rcut,/*cutoff distance*/
			   Box box/*Contains information and methods about the box, like apply_pbc*/
			   ):
	  v(v), Mv(Mv), FandG(FandG), box(box){
	  rcut2 = (rcut*rcut);

	}

	/*Start with Mv[i] = 0*/
	inline __device__ computeType zero(){ return computeType();}

	/*Get element pi from v*/
	inline __device__ infoType getInfo(int pi){
	  return make_real3(v[pi]); /*Works for realX */
	}
	/*Compute the dot product Mr_ij(3x3)·vj(3)*/
	inline __device__ computeType compute(const real4 &pi, const real4 &pj,
					      const infoType &vi, const infoType &vj){
	  /*Distance between the pair inside the primary box*/
	  real3 rij = box.apply_pbc(make_real3(pj)-make_real3(pi));

	  const real r2 = dot(rij, rij);

	  //The table takes care of this case
	  //if(r2 >= rcut2) return computeType();

	  /*Fetch RPY coefficients from a table, see RPYPSE_near*/
	  /* Mreal(r) = (F(r)·I + (G(r)-F(r))·rr)/(6*pi*vis*a) */
	  //f and g are divided by 6*pi*vis*a in the texture
#if CUB_PTX_ARCH < 300
	  constexpr auto cubModifier = cub::LOAD_DEFAULT;
#else
	  constexpr auto cubModifier = cub::LOAD_LDG;
#endif
	  const real2 fg = FandG.get<cubModifier>(sqrt(r2));
	  const real f = fg.x;
	  const real g = fg.y;
	  /*If i==j */
	  if(r2==real(0.0)){
	    /*M_ii·vi = F(0)*I·vi/(6*pi*vis*a) */
	    return f*make_real3(vj);
	  }
	  /*Update the result with Mr_ij·vj, the current box dot the current three elements of v*/
	  /*This expression is a little obfuscated, Mr_ij·vj*/
	  /*
	    Mr = (f(r)*I+(g(r)-f(r))*r(diadic)r)/(6*pi*vis*a) - > (M·v)_ß = (f(r)·v_ß + (g(r)-f(r))·v·(r(diadic)r))/(6*pi*vis*a)
	    Where f and g are the RPY coefficients, which are already divided by 6*pi*vis*a in the table.
	  */
	  const real invr2 = real(1.0)/r2;
	  const real gmfv = (g-f)*dot(rij, vj)*invr2;
	  /*gmfv = (g(r)-f(r))·( vx·rx + vy·ry + vz·rz )*/
	  /*((g(r)-f(r))·v·(r(diadic)r) )_ß = gmfv·r_ß*/
	  return make_real3(f*vj + gmfv*rij);
	}
	/*Just sum the result of each interaction*/
	inline __device__ void accumulate(computeType &total, const computeType &cur){total += cur;}

	/*Write the final result to global memory*/
	inline __device__ void set(int id, const computeType &total){
	  Mv[id] += make_real3(total);
	}
	vtype* v;
	real3* Mv;
	TabulatedFunction<real2> FandG;
	real rcut2;
	Box box;
      };
    }

    /*Compute Mr·v*/
    template<typename vtype>
    void PSE::Mdot_near(real3 *Mv, vtype *v, cudaStream_t st){
      /*Near contribution*/
      sys->log<System::DEBUG1>("[BDHI::PSE] Computing MF real space...");
      /*Create the Transverser struct*/
      PSE_ns::RPYNearTransverser<vtype> tr(v, Mv,
					   *RPY_near,
					   rcut, box);
      /*Transvese using tr*/
      cl->transverseListWithNeighbourList(tr, st);

    }

    namespace PSE_ns{

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
      /*** Kernels to compute the wave space contribution of M·F and B·dW *******/


      /*Apply the projection operator to a wave number with a certain complex factor.
	res = (I-k^k)·factor
	See i.e eq. 16 in [1].
       */
      inline __device__ cufftComplex3 projectFourier(const real3 &k, const cufftComplex3 &factor){

	//const real k2 = dot(k,k);
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


      /*Spreads the 3D quantity v (i.e the force) to a regular grid given by utils
	For that it uses a Peskin Gaussian kernel of the form f(r) = prefactor·exp(-tau·r^2), see [2].
	Applies the operator S in [1].
	Launch a block per particle.
      */
      template<typename vtype> /*Can take a real3 or a real4*/
      __global__ void particles2GridD(real4 * __restrict__ pos, /*Particle positions*/
				      vtype * __restrict__ v,   /*Per particle quantity to spread*/
				      real3 * __restrict__ gridVels, /*Interpolated values, size ncells*/
				      int N, /*Number of particles*/
				      int3 P, /*Gaussian kernel support in each dimension*/
				      Grid grid, /*Grid information and methods*/
				      real3 prefactor,/*Prefactor for the kernel, (2*xi*xi/(pi·eta))^3/2*/
				      real3 tau /*Kernel exponential factor, 2*xi*xi/eta*/){
	const int id = blockIdx.x;
	const int tid = threadIdx.x;
	if(id>=N) return;

	/*Get pos and v (i.e force)*/
	__shared__ real3 pi;
	__shared__ real3 vi_pf;
	__shared__ int3 celli;
	if(tid==0){
	  pi = make_real3(pos[id]);
	  vi_pf = make_real3(v[id])*prefactor;
	  /*Get my cell*/
	  celli = grid.getCell(pi);
	}
	/*Conversion between cell number and cell center position*/
	const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);
	const int3 supportCells = 2*P + 1;
	const int numberNeighbourCells = supportCells.x*supportCells.y*supportCells.z;

	__syncthreads();
	for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	  /*Compute neighbouring cell*/
	  int3 cellj = make_int3(celli.x + i%supportCells.x - P.x,
				 celli.y + (i/supportCells.x)%supportCells.y - P.y,
				 celli.z + i/(supportCells.x*supportCells.y) - P.z );
	  cellj = grid.pbc_cell(cellj);

	  /*Distance from particle i to center of cell j*/
	  const real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize-cellPosOffset);
	  const real r2 = dot(rij, rij);

	  /*The weight of particle i on cell j*/
	  const real3 weight = vi_pf*make_real3(exp(-r2*tau.x), exp(-r2*tau.y), exp(-r2*tau.z));

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
	also adds stochastic fourier noise, see addBrownianNoise
	 Input: gridForces = FFTf·S·F
	 Output:gridVels = B·FFTf·S·F + 1/√σ·√B·dWw
	 See sec. B.2 in [1]
       */
      /*A thread per fourier node*/
      __global__ void forceFourier2Vel(cufftComplex3 * gridForces, /*Input array*/
				       cufftComplex3 * gridVels, /*Output array, can be the same as input*/
				       /*Fourier scaling factors, see PSE_ns::fillFourierScaling Factors*/
				       const real3* Bfactor,
				       Grid grid/*Grid information and methods*/
				       ){
	/*Get my cell*/
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;
	/*Only the first half of the innermost dimension is stored, the rest is redundant*/
	if(cell.x>=grid.cellDim.x/2+1) return;
	if(cell.y>=grid.cellDim.y) return;
	if(cell.z>=grid.cellDim.z) return;

	const int icell = grid.getCellIndex(cell);
	if(icell == 0){
	  gridVels[0] = {0,0, 0,0, 0,0};
	  return;
	}

	const real3 k = cellToWaveNumber(cell, grid.cellDim, grid.box.boxSize);

	/*Get my scaling factor B(k,xi,eta)*/
	const real3 B = Bfactor[icell];
	cufftComplex3 factor = gridForces[icell];

	factor.x *= B.x;
	factor.y *= B.y;
	factor.z *= B.z;

	/*Store vel in global memory, note that this is overwritting any previous value in gridVels*/
	gridVels[icell] = projectFourier(k, factor);
      }

      /*Computes the long range stochastic velocity term
	Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
	= σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	See sec. B.2 in [1]
	This kernel gets v_k = gridVelsFourier = B·FFtt·S·F as input and adds 1/√σ·√B(k)·dWw.
	Keeping special care that v_k = v*_{N-k}, which implies that dWw_k = dWw*_{N-k}

	Launch a thread per cell grid/fourier node
      */
      __global__ void fourierBrownianNoise(/*Values of vels on each cell*/
					   cufftComplex3 *__restrict__ gridVelsFourier,
					   /*Fourier scaling factors, see PSE_ns::fillFourierScalingFactor*/
					   const real3* __restrict__ Bfactor,
					   Grid grid, /*Grid parameters. Size of a cell, number of cells...*/
					   real prefactor,/* sqrt(2·T/dt)*/
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

	  const real3 Bsq = sqrt(Bfactor[icell]);

	  cufftComplex3 factor = noise;
	  factor.x *= Bsq.x;
	  factor.y *= Bsq.y;
	  factor.z *= Bsq.z;

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

	  const real3 Bsq = sqrt(Bfactor[icell]);
	  cufftComplex3 factor = noise;
	  /*v_{N-k} = v*_k, so the complex noise must be conjugated*/
	  factor.x.y *= real(-1.0);
	  factor.y.y *= real(-1.0);
	  factor.z.y *= real(-1.0);

	  factor.x *= Bsq.x;
	  factor.y *= Bsq.y;
	  factor.z *= Bsq.z;

	  gridVelsFourier[icell] += projectFourier(k, factor);
	}
      }


      /*Interpolates a quantity (i.e velocity) from its values in the grid to the particles.
	For that it uses a Gaussian kernel of the form f(r) = prefactor·exp(-tau·r^2)
	Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
	= σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)

	Input: gridVels = FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
	Output: Mv = σ·St·gridVels
	The first term is computed in forceFourier2Vel and the second in fourierBrownianNoise
      */
      template<typename vtype>
      __global__ void grid2ParticlesD(real4 * __restrict__ pos,
				      vtype * __restrict__ Mv, /*Result (i.e Mw·F)*/
				      real3 * __restrict__ gridVels, /*Values in the grid*/
				      int N, /*Number of particles*/
				      int3 P, /*Gaussian kernel support in each dimension*/
				      Grid grid, /*Grid information and methods*/
				      real3 prefactor,/*Prefactor for the kernel, (2*xi*xi/(pi·eta))^3/2*/
				      real3 tau /*Kernel exponential factor, 2*xi*xi/eta*/
				      ){
	/*A thread per particle */
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=N) return;
	/*Get my particle and my cell*/

	const real3 pi = make_real3(pos[id]);
	const int3 celli = grid.getCell(pi);
	int3 cellj;
	/*The S^T = St = σ S*/
	prefactor *= (grid.cellSize.x*grid.cellSize.y*grid.cellSize.z);

	real3  result = make_real3(0);

	int x,y,z;
	/*Transform cell number to cell center position*/
	real3 cellPosOffset = real(0.5)*(grid.cellSize-grid.box.boxSize);
	/*Transvers the Pth neighbour cells*/
	for(z=-P.z; z<=P.z; z++){
	  cellj.z = grid.pbc_cell_coord<2>(celli.z + z);
	  for(y=-P.y; y<=P.y; y++){
	    cellj.y = grid.pbc_cell_coord<1>(celli.y + y);
	    for(x=-P.x; x<=P.x; x++){
	      cellj.x = grid.pbc_cell_coord<0>(celli.x + x);
	      /*Get neighbour cell*/
	      int jcell = grid.getCellIndex(cellj);

	      /*Compute distance to center*/
	      real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize - cellPosOffset);
	      real r2 = dot(rij, rij);
	      /*Interpolate cell value and sum*/
	      real3 cellj_vel = make_real3(gridVels[jcell]);
	      result += prefactor*make_real3(exp(-tau.x*r2), exp(-tau.y*r2), exp(-tau.z*r2))*cellj_vel;
	    }
	  }
	}
	/*Write total to global memory*/
	Mv[id] += result;
      }

    }

    /*Far contribution of M·F and B·dW, see begining of file

      Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw =
      = σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
    */
    template<typename vtype>
    void PSE::Mdot_far(real3 *Mv, vtype *v, cudaStream_t st){

      sys->log<System::DEBUG1>("[BDHI::PSE] Computing MF wave space....");
      int numberParticles = pg->getNumberParticles();
      auto pos = pd->getPos(access::location::gpu, access::mode::read);

      cufftSetStream(cufft_plan_forward, st);
      cufftSetStream(cufft_plan_inverse, st);

      /*Clean gridVels*/
      int ncells = grid.cellDim.x*grid.cellDim.y*grid.cellDim.z;
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<ncells?BLOCKSIZE:ncells;
      int Nblocks  =  ncells/Nthreads +  ((ncells%Nthreads!=0)?1:0);

      sys->log<System::DEBUG2>("[BDHI::PSE] Setting vels to zero...");
      //Note that the same storage space is used for Fourier and real space
      //The real space is the only one that needs to be cleared.
      auto d_gridVels = (real3*)thrust::raw_pointer_cast(gridVelsFourier.data());
      fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(d_gridVels,
						make_real3(0), ncells);
      auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());


      Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);

      /*Gaussian spreading/interpolation kernel parameters, s(r) = prefactor*exp(-tau*r2)
	See eq. 13 in [2]
      */
      real3 prefactorGaussian = pow(2.0*psi*psi/(M_PI), 1.5)/make_real3(pow(eta.x,1.5), pow(eta.y, 1.5), pow(eta.z, 1.5));
      real3 tau       = 2.0*psi*psi/eta;
      sys->log<System::DEBUG2>("[BDHI::PSE] Particles to grid");
      /*Spread force on particles to grid positions -> S·F*/
      //Launch a small block per particle
      {
	int3 support = 2*P+1;
	int threadsPerParticle = 64;
	int numberNeighbourCells = support.x*support.y*support.z;
	if(numberNeighbourCells < 64) threadsPerParticle = 32;

	PSE_ns::particles2GridD<<<numberParticles, threadsPerParticle, 0, st>>>
	  (pos.raw(), v, d_gridVels, numberParticles, P, grid, prefactorGaussian, tau);
      }

      sys->log<System::DEBUG2>("[BDHI::PSE] Taking grid to wave space");
      /*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
      auto cufftStatus =
	cufftExecR2C(cufft_plan_forward,
		     (cufftReal*)d_gridVels,
		     (cufftComplex*)d_gridVelsFourier);
      if(cufftStatus != CUFFT_SUCCESS){
	sys->log<System::CRITICAL>("[BDHI::PSE] Error in forward CUFFT");
      }

      sys->log<System::DEBUG2>("[BDHI::PSE] Wave space velocity scaling");
      if(temperature > real(0.0))
	sys->log<System::DEBUG2>("[BDHI::PSE] Wave space brownian noise");

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
      //B in [1]
      auto d_fourierFactor = thrust::raw_pointer_cast(fourierFactor.data());

      PSE_ns::forceFourier2Vel<<<NblocksCells, NthreadsCells, 0, st>>>
      ((PSE_ns::cufftComplex3*) d_gridVelsFourier, //Input: FFT·S·F
       (PSE_ns::cufftComplex3*) d_gridVelsFourier, //Output: B·FFT·S·F
       d_fourierFactor, //B
       grid);
      //eq 19 and beyond in [1].
      //The sqrt(2*T/dt) factor needs to be here because far noise is summed to the M·F term.
      /*Add the stochastic noise to the fourier velocities if T>0 -> 1/√σ·√B·dWw */
      if(temperature > real(0.0)){
	static ullint counter = 0; //Seed the rng differently each call
	counter++;
	sys->log<System::DEBUG2>("[BDHI::PSE] Wave space brownian noise");

	real prefactor = sqrt(2*temperature/dt/(grid.cellSize.x*grid.cellSize.y*grid.cellSize.z));

	PSE_ns::fourierBrownianNoise<<<NblocksCells, NthreadsCells, 0, st>>>(
	            		    //In: B·FFT·S·F -> Out: B·FFT·S·F + 1/√σ·√B·dWw
 			            (PSE_ns::cufftComplex3*)d_gridVelsFourier,
				    d_fourierFactor, //B
				    grid,
				    prefactor, // 1/√σ· sqrt(2*T/dt),
				    seed, //Saru needs two seeds apart from thread id
				    counter);
      }

      sys->log<System::DEBUG2>("[BDHI::PSE] Going back to real space");
      /*Take the fourier velocities back to real space ->  FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
      cufftStatus =
	cufftExecC2R(cufft_plan_inverse,
		     (cufftComplex*)d_gridVelsFourier,
		     (cufftReal*)d_gridVels);
      if(cufftStatus != CUFFT_SUCCESS){
	sys->log<System::CRITICAL>("[BDHI::PSE] Error in inverse CUFFT");
      }

      sys->log<System::DEBUG2>("[BDHI::PSE] Grid to particles");
      /*Interpolate the real space velocities back to the particle positions ->
	Output: Mv = Mw·F + sqrt(2*T/dt)·√Mw·dWw = σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
      PSE_ns::grid2ParticlesD<<<Nblocks, Nthreads, 0, st>>>
	(pos.raw(), Mv, d_gridVels,
	 numberParticles, P, grid, prefactorGaussian, tau);

      sys->log<System::DEBUG2>("[BDHI::PSE] MF wave space Done");

    }

    void PSE::computeMF(real3* MF, cudaStream_t st){
      sys->log<System::DEBUG1>("[BDHI::PSE] Computing MF....");
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      Mdot<real4>(MF, force.raw(), st);
    }

    namespace PSE_ns{
      /*LanczosAlgorithm needs a functor that computes the product M·v*/
      /*Dotctor takes a list transverser and a cell list on construction,
	and the operator () takes an array v and returns the product M·v*/
      struct Dotctor{
	/*Dotctor uses the same transverser as in Mr·F*/
	typedef typename PSE_ns::RPYNearTransverser<real3> myTransverser;
	myTransverser Mv_tr;
	shared_ptr<CellList> cl;
	int numberParticles;
	cudaStream_t st;

	Dotctor(myTransverser Mv_tr, shared_ptr<CellList> cl, int numberParticles, cudaStream_t st):
	  Mv_tr(Mv_tr), cl(cl), numberParticles(numberParticles), st(st){ }

	inline void operator()(real3* Mv, real3 *v){
	  /*Update the transverser input and output arrays*/
	  Mv_tr.v = v;
	  Mv_tr.Mv = Mv;
	  int BLOCKSIZE = 128;
	  int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
	  int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
	  fillWithGPU<<<Nblocks, Nthreads, 0 ,st>>>(Mv, make_real3(0), numberParticles);
	  /*Perform the dot product*/
	  cl->transverseListWithNeighbourList(Mv_tr, st);

	}
      };
    }

    void PSE::computeBdW(real3* BdW, cudaStream_t st){

      sys->log<System::DEBUG2>("[BDHI::PSE] Real space brownian noise");
      /*Far contribution is in Mdot_far*/
      /*Compute stochastic term only if T>0 */
      if(temperature == real(0.0)) return;
      int numberParticles = pg->getNumberParticles();

      /*List transverser for near dot product*/
      PSE_ns::RPYNearTransverser<real3> tr(nullptr, nullptr,
					   *RPY_near,
					   rcut, box);
      /*Functor for dot product*/
      PSE_ns::Dotctor Mvdot_near(tr, cl, numberParticles, st);
      /*Lanczos algorithm to compute M_near^1/2 · noise. See LanczosAlgorithm.cuh*/
      real *noise = lanczos->getV(numberParticles);
      curandGenerateNormal(curng, noise,
			   3*numberParticles + (3*numberParticles)%2,
			   real(0.0), real(1.0));

      auto status = lanczos->solve(Mvdot_near,
				   (real *)BdW, noise,
				   numberParticles,
				   lanczosTolerance);

      if(status == LanczosStatus::TOO_MANY_ITERATIONS){
	sys->log<System::WARNING>("[BDHI::PSE] This is probably fine, but Lanczos could not achieve convergence, try increasing the tolerance or switching to double precision.");
      }
      else if(status != LanczosStatus::SUCCESS){
	sys->log<System::CRITICAL>("[BDHI::PSE] Lanczos Algorithm failed with code %d!", status);
      }

    }

    void PSE::computeDivM(real3* divM, cudaStream_t st){}


    void PSE::finish_step(cudaStream_t st){
      sys->log<System::DEBUG2>("[BDHI::PSE] Finishing step");

    }
  }
}

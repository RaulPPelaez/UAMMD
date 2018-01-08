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
 
References:

[1]  Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations
           -  https://arxiv.org/pdf/1611.09322.pdf
[2]  Spectral accuracy in fast Ewald-based methods for particle simulations
           -  http://www.sciencedirect.com/science/article/pii/S0021999111005092

TODO: 
100- Treat near textures as table potentials, manually interpolate using PotentialTable
100- Use native cufft memory layout

Special thanks to Marc Melendez and Florencio Balboa.
*/
#include"BDHI_PSE.cuh"
#include"utils/GPUUtils.cuh"
#include"misc/TabulatedFunction.cuh"
#include"RPY_PSE.cuh"
#include<vector>
#include<algorithm>

namespace uammd{
  namespace BDHI{

    namespace PSE_ns{
      /* Initialize the cuRand states for later use in PSE_ns::fourierBrownianNoise */
      __global__ void initCurand(curandState *states, ullint seed, int size){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>= size) return;
	/*Each state has the same seed and a different sequence, this makes 
	  initialization really slow, but ensures good random number properties*/
	//curand_init(seed, id, 0, &states[id]);
	/*Faster but bad random number properties*/
	curand_init((seed<<20)+id, 0, 0, &states[id]);   
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
	double3 pi2invL = 2.0*double(M_PI)/make_double3(grid.box.boxSize);

	/*Get my wave number*/
	double3 K = make_double3(cell)*pi2invL;
	/*Remember that FFT stores wave numbers as K=0:N/2+1:-N/2:-1 */    
	if(cell.x >= (grid.cellDim.x+1)/2) K.x -= double(grid.cellDim.x)*pi2invL.x;
	if(cell.y >= (grid.cellDim.y+1)/2) K.y -= double(grid.cellDim.y)*pi2invL.y;
	if(cell.z >= (grid.cellDim.z+1)/2) K.z -= double(grid.cellDim.z)*pi2invL.z;

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
	     BDHI::Parameters par):
      pd(pd), pg(pg), sys(sys),
      dt(par.dt),
      temperature(par.temperature),
      box(par.box), grid(box, int3()),
      psi(1.5){
      if(box.boxSize.x == real(0.0) && box.boxSize.y == real(0.0) && box.boxSize.z == real(0.0)){
	sys->log<System::CRITICAL>("[BDHI::PSE] Box of size zero detected, cannot work without a box! (make sure a box parameter was passed)");
      }
      this->lanczosTolerance = 0.05;
      this->lanczos = std::make_shared<LanczosAlgorithm>(sys, lanczosTolerance);
  
      
      sys->log<System::MESSAGE>("[BDHI::PSE] Initialized");

      int numberParticles = pg->getNumberParticles();

  
      /* M = Mr + Mw */
      sys->log<System::MESSAGE>("[BDHI::PSE] Self mobility: %f", 1.0/(6*M_PI*par.viscosity*par.hydrodynamicRadius));
  
      /*Compute M0*/
      double pi = M_PI;
      double a = par.hydrodynamicRadius;
      double prefac = (1.0/(24*sqrt(pi*pi*pi)*psi*a*a*par.viscosity));
      /*M0 = Mr(0) = F(0)(I-r^r) + G(0)(r^r) = F(0) = Mii_r . 
	See eq. 14 in [1] and RPYPSE_nearTextures*/
      this->M0 = prefac*(1-exp(-4*a*a*psi*psi)+4*sqrt(pi)*a*psi*std::erfc(2*a*psi));  

      /****Initialize near space part: Mr *******/
      real er = par.tolerance; /*Short range error tolerance*/
      /*Near neighbour list cutoff distance, see sec II:C in [1]*/
      rcut = a*sqrt(-log(er))/psi;

      /*Initialize the neighbour list */
      this->cl = std::make_shared<CellList>(pd, pg, sys);
  
      /*Initialize the near RPY textures*/
      RPYPSE_near rpy(par.viscosity, par.hydrodynamicRadius, psi, M0, rcut);

      
      int nPointsTable = 4096;
      tableDataRPY.resize(nPointsTable+1);
      RPY_near = std::make_shared<TabulatedFunction<real2>>(thrust::raw_pointer_cast(tableDataRPY.data()),
							    nPointsTable, 0.0, rcut*rcut, rpy);
      
      /****Initialize wave space part: Mw ******/
      real ew = par.tolerance; /*Long range error tolerance*/
      /*Maximum wave number for the far calculation*/
      kcut = 2*psi*sqrt(-log(ew))/a;
      /*Corresponding real space grid size*/
      double hgrid = 2*pi/kcut;

      /*Create a grid with cellDim cells*/
      /*This object will contain useful information about the grid,
	mainly number of cells, cell size and usual parameters for
	using it in the gpu, as invCellSize*/  
      int3 cellDim = make_int3(box.boxSize/hgrid)+1;

      /*FFT likes a number of cells as cellDim.i = 2^n·3^l·5^m */
      {
	int* cdim = &cellDim.x;
	/*Store up to 2^5·3^5·5^5 in tmp*/    
	int n= 5;
	std::vector<int> tmp(n*n*n, 0);
	int max_dim = *std::max_element(cdim, cdim+2);
    
	do{
	  tmp.resize(n*n*n, 0);
	  fori(0,n)forj(0,n)for(int k=0; k<n;k++)
	    tmp[i+n*j+n*n*k] = pow(2,i)*pow(3,j)*pow(5,k);
	  n++;
	  /*Sort this array in ascending order*/
	  std::sort(tmp.begin(), tmp.end());      
	}while(tmp.back()<max_dim); /*if n=5 is not enough, include more*/
	int i=0;
	/*Now look for the nearest value in tmp that is greater than each cell dimension*/
	forj(0,3){
	  i = 0;
	  while(tmp[i]<cdim[j]) i++;
	  cdim[j] = tmp[i];
	}
      }

      /*Store grid parameters in a Mesh object*/
      grid = Grid(box, cellDim);

      /*Print information*/
      sys->log<System::MESSAGE>("[BDHI::PSE] Box Size: %f %f %f", box.boxSize.x, box.boxSize.y, box.boxSize.z);
      sys->log<System::MESSAGE>("[BDHI::PSE] Splitting factor: %f",psi);  
      sys->log<System::MESSAGE>("[BDHI::PSE] Close range distance cut off: %f",rcut);
      int3 cDSR = make_int3(box.boxSize/rcut + 0.5);
      sys->log<System::MESSAGE>("[BDHI::PSE] Close range grid size: %d %d %d", cDSR.x, cDSR.y, cDSR.z);

      sys->log<System::MESSAGE>("[BDHI::PSE] Far range wave number cut off: %f", kcut);
      sys->log<System::MESSAGE>("[BDHI::PSE] Far range grid size: %d %d %d", cellDim.x, cellDim.y, cellDim.z);
  
      cudaStreamCreate(&stream);
      cudaStreamCreate(&stream2);
  
      /*The quantity spreaded to the grid in real space*/
      /*The layout of this array is
	fx000, fy000, fz000, fx001, fy001, fz001..., fxnnn, fynnn, fznnn. n=ncells-1*/
      /*Can be Force when spreading particles to the grid and
	velocities when interpolating from the grid to the particles*/
      int ncells = grid.cellDim.x*grid.cellDim.y*grid.cellDim.z;
      gridVels.resize(ncells, real3());
      /*Same but in wave space, in MF it will be the forces that get transformed to velocities*/
      /*3 complex numbers per cell*/
      gridVelsFourier.resize(3*ncells, cufftComplex());

      /*Initialize far stochastic random generators*/
      if(temperature > real(0.0)){
	sys->log<System::DEBUG>("[BDHI::PSE] Initializing cuRand....");
	int fnsize = ncells;
	fnsize += fnsize%2;
	farNoise.resize(fnsize);
	auto d_farNoise = thrust::raw_pointer_cast(farNoise.data());
	PSE_ns::initCurand<<<fnsize/32+1, 32,0, stream>>>(d_farNoise, sys->rng().next(), fnsize);
      }
      /*Grid spreading/interpolation parameters*/
      /*Gaussian spreading/interpolation kernel support points neighbour distance
	See sec. 2.1 in [2]*/
      this->P = make_int3(1);
      double3 pw = make_double3(2*P+1);/*Number of support points*/
      double3 h = make_double3(grid.cellSize); /*Cell size*/
      /*Number of standard deviations in the grid's Gaussian kernel support*/
      double3 m = 0.976*sqrt(M_PI)*make_double3(sqrt(pw.x), sqrt(pw.y), sqrt(pw.z));
      /*Standard deviation of the Gaussian kernel*/
      double3 w   = pw*h/2.0;
      /*Gaussian splitting parameter*/
      this->eta = make_real3(pow(2.0*psi, 2)*w*w/(m*m));

      /*B in [1], this array stores, for each cell/fourier node,
	the scaling factor to go from forces to velocities in fourier space*/
      fourierFactor.resize(ncells);

      int BLOCKSIZE = 128;      
      /*Launch a thread per cell/node*/
      dim3 NthreadsCells = BLOCKSIZE;
      dim3 NblocksCells;
      NblocksCells. x=  grid.cellDim.x/NthreadsCells.x + 1;
      NblocksCells. y=  grid.cellDim.y/NthreadsCells.y + 1;
      NblocksCells. z=  grid.cellDim.z/NthreadsCells.z + 1;

      PSE_ns::fillFourierScalingFactor<<<NblocksCells, NthreadsCells, 0, stream2>>>
	(thrust::raw_pointer_cast(fourierFactor.data()), grid,
	 par.hydrodynamicRadius, par.viscosity, psi, eta);
        

      /*Set up cuFFT*/

      /*I will be handling workspace memory*/
      cufftSetAutoAllocation(cufft_plan_forward, 0);
      cufftSetAutoAllocation(cufft_plan_inverse, 0);

      int3 cdtmp = {grid.cellDim.z, grid.cellDim.y, grid.cellDim.x};
      /*I want to make three 3D FFTs, each one using one of the three interleaved coordinates*/
      auto cufftStatus = cufftPlanMany(&cufft_plan_forward,
				       3, &cdtmp.x, /*Three dimensional FFT*/
				       &cdtmp.x,
				       /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
				       3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
				       /*Same format in the output*/
				       &cdtmp.x,
				       3, 1,
				       /*Perform 3 Batched FFTs*/
				       CUFFT_R2C, 3);

      if(cufftStatus != CUFFT_SUCCESS){
	sys->log<System::CRITICAL>("[BDHI::PSE] Problem setting up cuFFT Forward!");
      }
      sys->log<System::DEBUG>("[BDHI::PSE] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
      /*Same as above, but with C2R for inverse FFT*/
      cufftStatus = cufftPlanMany(&cufft_plan_inverse,
				  3, &cdtmp.x, /*Three dimensional FFT*/
				  &cdtmp.x,
				  /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
				  3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
				  &cdtmp.x,
				  3, 1,
				  /*Perform 3 FFTs*/
				  CUFFT_C2R, 3);

      if(cufftStatus != CUFFT_SUCCESS){
	sys->log<System::CRITICAL>("[BDHI::PSE] Problem setting up cuFFT Inverse!");
      }

      /*Allocate cuFFT work area*/
      size_t cufftWorkSizef = 0, cufftWorkSizei = 0;

      cufftGetSize(cufft_plan_forward, &cufftWorkSizef);
      cufftGetSize(cufft_plan_inverse, &cufftWorkSizei);

      size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei);

      sys->log<System::DEBUG>("[BDHI::PSE] Necessary work space for cuFFT: %s", printUtils::prettySize(cufftWorkSize).c_str());
      size_t free_mem, total_mem;
      cudaMemGetInfo(&free_mem, &total_mem);

      if(free_mem<cufftWorkSize){
	sys->log<System::CRITICAL>("[BDHI::PSE] Not enough memory in device to allocate cuFFT!!, try lowering the splitting parameter!");
      }

      cufftWorkArea.resize(cufftWorkSize/sizeof(real)+1);
      auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      
      cufftSetWorkArea(cufft_plan_forward, (void*)d_cufftWorkArea);
      cufftSetWorkArea(cufft_plan_inverse, (void*)d_cufftWorkArea);


      //Init rng
      curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT);
    
      curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next());

      thrust::device_vector<real> noise(30000);
      auto noise_ptr = thrust::raw_pointer_cast(noise.data());
      //Warm cuRNG
      curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0);
      curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0);

  
      cudaDeviceSynchronize();
    }
  


    PSE::~PSE(){
      cufftDestroy(cufft_plan_inverse);
      cufftDestroy(cufft_plan_forward);
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
  
      Mdot_near<vtype>(Mv, v, st);
      //Mdot_farThread = std::thread(&PSE::Mdot_far<vtype>, this, Mv, v, stream2);
      Mdot_far<vtype>(Mv, v, stream2);

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
			   real M0, /*RPY_near(0)*/
			   real rcut,/*cutoff distance*/
			   Box box/*Contains information and methods about the box, like apply_pbc*/
			   ):
	  v(v), Mv(Mv), FandG(FandG), M0(M0), rcut(rcut), box(box){
	  rcut2 = (rcut*rcut);
	}
    
	/*Start with Mv[i] = 0*/
	inline __device__ computeType zero(){ return make_real3(0);}

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
	  /*If i==j */
	  if(r2==real(0.0)){
	    /*M_ii·vi = M0*I·vi */
	    return vj;
	  }
	  if(r2 > rcut2) return make_real3(0);


      
	  /*M0 := Mii := RPY_near(0) is multiplied once at the end*/
	  /*Fetch RPY coefficients from a table, see RPYPSE_near*/
	  /* Mreal(r) = M0*(F(r)·I + (G(r)-F(r))·rr) */
      
	  const real2 fg = FandG(r2);
	  const real f = fg.x;
	  const real g = fg.y;
	  

	  /*Update the result with Mr_ij·vj, the current box dot the current three elements of v*/
	  /*This expression is a little obfuscated, Mr_ij·vj*/
	  /*
	    Mr = f(r)*I+(g(r)-f(r))*r(diadic)r - > (M·v)_ß = f(r)·v_ß + (g(r)-f(r))·v·(r(diadic)r)
	    Where f and g are the RPY coefficients
	  */
	  const real invr2 = real(1.0)/r2;
	  const real gmfv = (g-f)*dot(rij, vj)*invr2;
	  /*gmfv = (g(r)-f(r))·( vx·rx + vy·ry + vz·rz )*/
	  /*((g(r)-f(r))·v·(r(diadic)r) )_ß = gmfv·r_ß*/
	  return (f*vj + gmfv*rij);
	}
	/*Just sum the result of each interaction*/
	inline __device__ void accumulate(computeType &total, const computeType &cur){total += cur;}

	/*Write the final result to global memory*/
	inline __device__ void set(int id, const computeType &total){
	  Mv[id] += M0*total;
	}
	vtype* v;
	real3* Mv;    
	real M0;
	TabulatedFunction<real2> FandG;
	real rcut, rcut2;
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
					   M0, rcut, box);
      /*Update the list if needed*/
      cl->updateNeighbourList(box, rcut, st);
      /*Transvese using tr*/
      cl->transverseList(tr, st);

    }

#define TPP 1
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

      /*Spreads the 3D quantity v (i.e the force) to a regular grid given by utils
	For that it uses a Gaussian kernel of the form f(r) = prefactor·exp(-tau·r^2)
      */
      template<typename vtype> /*Can take a real3 or a real4*/
      __global__ void particles2GridD(real4 * __restrict__ pos,
				      vtype * __restrict__ v,
				      real3 * __restrict__ gridVels, /*Interpolated values, size ncells*/
				      int N, /*Number of particles*/
				      int3 P, /*Gaussian kernel support in each dimension*/
				      Grid grid, /*Grid information and methods*/
				      real3 prefactor,/*Prefactor for the kernel, (2*xi*xi/(pi·eta))^3/2*/
				      real3 tau /*Kernel exponential factor, 2*xi*xi/eta*/
				      ){
	/*TPP threads per particle*/
	int offset = threadIdx.x%TPP;
	int id = blockIdx.x*blockDim.x + threadIdx.x/TPP;
	if(id>=N) return;

	/*Get pos and v (i.e force)*/
	real3 pi = make_real3(pos[id]);
	real3 vi_pf = make_real3(v[id])*prefactor;

	/*Get my cell*/
	int3 celli = grid.getCell(pi);
    
	int3 cellj;    
	int x,y,z;
	/*Conversion between cell number and cell center position*/
	real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);

	/*Transverse the Pth neighbour cells*/
	for(x=-P.x+offset; x<=P.x; x+=TPP) 
	  for(z=-P.z; z<=P.z; z++)
	    for(y=-P.y; y<=P.y; y++){
	      /*Get the other cell*/
	      /*Corrected with PBC*/
	      cellj = grid.pbc_cell(celli+make_int3(x,y,z));	      	      

	      /*Get index of cell j*/
	      int jcell = grid.getCellIndex(cellj);
	      /*Distance from particle i to center of cell j*/
	      real3 rij = pi-make_real3(cellj)*grid.cellSize-cellPosOffset;
	      rij = grid.box.apply_pbc(rij);
	  
	      real r2 = dot(rij, rij);

	      /*The weight of particle i on cell j*/
	      real3 weight = vi_pf*make_real3(expf(-r2*tau.x), expf(-r2*tau.y), expf(-r2*tau.z));

	      /*Atomically sum my contribution to cell j*/
	      real* cellj_vel = &(gridVels[jcell].x);
	      atomicAdd(cellj_vel,   weight.x);
	      atomicAdd(cellj_vel+1, weight.y);
	      atomicAdd(cellj_vel+2, weight.z);
	    }
      }

      /*A convenient struct to pack 3 complex numbers, that is 6 real numbers*/
      struct cufftComplex3{
	cufftComplex x,y,z;
      };

      /*Scales fourier transformed forces in the regular grid to velocities*/
      /*A thread per cell*/
      __global__ void forceFourier2Vel(cufftComplex3 * gridForces, /*Input array*/
				       cufftComplex3 * gridVels, /*Output array, can be the same as input*/
				       real3* Bfactor, /*Fourier scaling factors, see PSE_ns::fillFourierScaling Factors*/ 
				       Grid grid/*Grid information and methods*/
				       ){
	/*Get my cell*/
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;
    
	if(cell.x>=grid.cellDim.x/2+1) return;
	if(cell.y>=grid.cellDim.y) return;
	if(cell.z>=grid.cellDim.z) return;
	int icell = grid.getCellIndex(cell);

	real3 pi2invL = real(2.0)*real(M_PI)/grid.box.boxSize;
	/*My wave number*/
	real3 k = make_real3(cell)*pi2invL;

	/*Be careful with the conjugates*/
	if(cell.x >= (grid.cellDim.x+1)/2) k.x -= real(grid.cellDim.x)*pi2invL.x;
	if(cell.y >= (grid.cellDim.y+1)/2) k.y -= real(grid.cellDim.y)*pi2invL.y;
	if(cell.z >= (grid.cellDim.z+1)/2) k.z -= real(grid.cellDim.z)*pi2invL.z;

	real k2 = dot(k,k);

	real kmod = sqrtf(k2);
	/*The node k=0 doesnt contribute, the checking is delayed to reduce thread divergence*/
	real invk2 = (icell==0)?real(0.0):real(1.0)/k2;

	/*Get my scaling factor B(k,xi,eta)*/
	real3 B = Bfactor[icell];
    
	cufftComplex3 fc = gridForces[icell];

	/*Compute V = B·(I-k^k)·F for both the real and complex part of the spreaded force*/
	real3 fr = make_real3(fc.x.x, fc.y.x, fc.z.x);

	real kfr = dot(k,fr)*invk2;
    
	real3 fi = make_real3(fc.x.y, fc.y.y, fc.z.y);
	real kfi = dot(k,fi)*invk2;

	real3 vr = (fr-k*kfr)*B;
	real3 vi = (fi-k*kfi)*B;
	/*Store vel in global memory*/
	gridVels[icell] = {vr.x, vi.x, vr.y, vi.y, vr.z, vi.z};    
      } 

      /*Interpolates a quantity (i.e velocity) from its values in the grid to the particles.
	For that it uses a Gaussian kernel of the form f(r) = prefactor·exp(-tau·r^2)
      */
      template<typename vtype>
      __global__ void grid2ParticlesD(real4 * __restrict__ pos,
				      vtype * __restrict__ Mv, /*Result (i.e M·F)*/
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
	      real3 cellj_vel = gridVels[jcell];
	      result += prefactor*make_real3(expf(-tau.x*r2), expf(-tau.y*r2), expf(-tau.z*r2))*cellj_vel;
	    }
	  }
	}
	/*Write total to global memory*/
	Mv[id] += result;
      }
      /*Computes the long range stochastic velocity term
	Mw·F + sqrt(Mw)·dWw = σ·St·FFTi·B·FFTf·S·F+ √σ·St·FFTi·√B·dWw = 
	= σ·St·FFTi( B·FFTf·S·F + 1/√σ·√B·dWw)
    
	This kernel gets gridVelsFourier = B·FFtt·S·F as input and adds 1/√σ·√B(k)·dWw.
    
	Launch a thread per cell grid/fourier node
      */
      __global__ void fourierBrownianNoise(
					   curandState_t * __restrict__ farNoise, /*cuRand generators*/
					   cufftComplex3 *__restrict__ gridVelsFourier, /*Values of vels on each cell*/
					   real3* __restrict__ Bfactor,/*Fourier scaling factors, see PSE_ns::fillFourierScalingFactor*/ 
					   Grid grid, /*Grid parameters. Size of a cell, number of cells...*/
					   real prefactor/* sqrt(2·T/dt) */
					   ){

	/*Get my cell*/
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;

	/*Get my cell index*/
	const int icell = grid.getCellIndex(cell);

	/*cuFFT R2C and C2R only store half of the innermost dimension, the one that varies the fastest
      
	  The input of R2C is real and the output of C2R is real. 
	  The only way for this to be true is if v_k={i,j,k} = v*_k{N-i, N-j, N-k}

	  So the conjugates are redundant and the is no need to compute them nor storage them.      
	*/
	/*Only compute the first half of the innermost dimension!*/
	if(cell.x>=grid.cellDim.x/2+1) return;
	if(cell.y>=grid.cellDim.y) return;
	if(cell.z>=grid.cellDim.z) return;
    
	/*K=0 is not added, no stochastic motion is added to the center of mass*/
	if(icell == 0){ return;}
	/*Fetch my rng*/
	curandState_t *rng = &farNoise[icell];

	/*Corresponding conjugate wave number,
	  as the innermost dimension conjugates are redundant, computing the conjugate is needed
	  only when cell.x == 0
      
	  Note that this line works even on nyquist points
	*/
	const int3 cell_conj = {(cell.x?(grid.cellDim.x-cell.x):cell.x),
				(cell.y?(grid.cellDim.y-cell.y):cell.y),
				(cell.z?(grid.cellDim.z-cell.z):cell.z)};    

    
	const int icell_conj = grid.getCellIndex(cell_conj);
    
	/*Compute the wave number of my cell and its conjugate*/
	const real3 pi2invL = real(2.0)*real(M_PI)/grid.box.boxSize;
    
	real3 k = make_real3(cell)*pi2invL;
	real3 kc = make_real3(cell_conj)*pi2invL;
    
	// /*Which is mirrored beyond cell_a = Ncells_a/2 */
	if(cell.x >= (grid.cellDim.x+1)/2) k.x -= real(grid.cellDim.x)*pi2invL.x;
	if(cell.y >= (grid.cellDim.y+1)/2) k.y -= real(grid.cellDim.y)*pi2invL.y;
	if(cell.z >= (grid.cellDim.z+1)/2) k.z -= real(grid.cellDim.z)*pi2invL.z;

	if(cell_conj.x >= (grid.cellDim.x+1)/2) kc.x -= real(grid.cellDim.x)*pi2invL.x;
	if(cell_conj.y >= (grid.cellDim.y+1)/2) kc.y -= real(grid.cellDim.y)*pi2invL.y;
	if(cell_conj.z >= (grid.cellDim.z+1)/2) kc.z -= real(grid.cellDim.z)*pi2invL.z;
    

	/*  Z = sqrt(B)· dW \propto (I-k^k)·dW */
	real k2 = dot(k,k);
	real invk2 = real(1.0)/k2;

	real3 Bsq_t = Bfactor[icell];
	real3 Bsq = {sqrtf(Bsq_t.x), sqrtf(Bsq_t.y), sqrtf(Bsq_t.z)};

	/*Compute gaussian complex noise, 
	  std = prefactor -> ||z||^2 = <x^2>/sqrt(2)+<y^2>/sqrt(2) = prefactor*/
	/*A complex random number for each direction*/
	cufftComplex3 vel;
	real complex_gaussian_sc = real(0.707106781186547)*prefactor; //1/sqrt(2)
	real2 tmp = curand_normal2(rng)*complex_gaussian_sc;
	vel.x.x = tmp.x;
	vel.x.y = tmp.y;
	tmp = curand_normal2(rng)*complex_gaussian_sc;
	vel.y.x = tmp.x;
	vel.y.y = tmp.y;
	tmp = curand_normal2(rng)*complex_gaussian_sc;
	vel.z.x = tmp.x;
	vel.z.y = tmp.y;
    
	bool nyquist = false;
	/*Beware of nyquist points!*/
	bool isXnyquist = (cell.x == grid.cellDim.x/2) && (grid.cellDim.x/2 == (grid.cellDim.x+1)/2);
	bool isYnyquist = (cell.y == grid.cellDim.y/2) && (grid.cellDim.y/2 == (grid.cellDim.y+1)/2);
	bool isZnyquist = (cell.z == grid.cellDim.z/2) && (grid.cellDim.z/2 == (grid.cellDim.z+1)/2);

	/*There are 8 nyquist points at most (cell=0,0,0 is excluded at the beggining)
	  These are the 8 vertex of the inferior left cuadrant. The O points:
               +--------+--------+
              /|       /|       /|
             / |      / |      / | 
            +--------+--------+  |
           /|  |    /|  |    /|  |
          / |  +---/-|--+---/-|--+
         +--------+--------+  |	/|
         |  |/ |  |  |/    |  |/ |
         |  O-----|--O-----|--+	 |
         | /|6 |  | /|7    | /|	 |
         |/ |  +--|/-|--+--|/-|--+
         O--------O--------+  |	/ 
         |5 |/    |4 |/    |  |/       
         |  O-----|--O-----|--+	 
     ^   | / 3    | / 2    | /  ^     
     |   |/       |/       |/  /     
     kz  O--------O--------+  ky
         kx ->     1

	*/
	if( (isXnyquist && cell.y==0   && cell.z==0)  || //1
	    (isXnyquist && isYnyquist  && cell.z==0)  || //2
	    (cell.x==0  && isYnyquist  && cell.z==0)  || //3
	    (isXnyquist && cell.y==0   && isZnyquist) || //4
	    (cell.x==0  && cell.y==0   && isZnyquist) || //5
	    (cell.x==0  && isYnyquist  && isZnyquist) || //6
	    (isYnyquist  && isYnyquist  && isZnyquist)   //7	       
	    ){
	  nyquist = true;

	  /*The random numbers are real in the nyquist points, 
	    ||r||^2 = <x^2> = ||Real{z}||^2 = <Real{z}^2>·sqrt(2) =  prefactor*/
	  real nqsc = real(1.41421356237310); //sqrt(2)
	  vel.x.x *= nqsc;
	  vel.x.y = 0;
	  vel.y.x *= nqsc;
	  vel.y.y = 0;
	  vel.z.x *= nqsc;
	  vel.z.y = 0;            
	}
	/*Z = bsq·(I-k^k)·vel*/
	/*Compute the dyadic product, both real and imaginary parts*/
	real3 f = make_real3(vel.x.x, vel.y.x, vel.z.x);
	real kf = dot(k,f)*invk2;
	real3 vr = (f-k*kf)*Bsq;
    
	f = make_real3(vel.x.y, vel.y.y, vel.z.y);    
	kf = dot(k,f)*invk2;
	real3 vi = (f-k*kf)*Bsq;

	/*Add the random velocities to global memory*/
	/*Velocities are stored as a complex number in each dimension, 
	  packed in a cufftComplex3 as 6 real numbers.
	  i.e three complex numbers one after the other*/
	cufftComplex3 kk = gridVelsFourier[icell];
	gridVelsFourier[icell] = {kk.x.x+vr.x, kk.x.y+vi.x,
				  kk.y.x+vr.y, kk.y.y+vi.y,
				  kk.z.x+vr.z, kk.z.y+vi.z};
    
	if(nyquist) return;
	/*Only if there is a conjugate point*/
    
	/*Z = bsq·(I-k^k)·vel*/
	/*Compute the dyadic product, both real and imaginary parts*/
	f = make_real3(vel.x.x, vel.y.x, vel.z.x);
	kf = dot(kc,f)*invk2;
	vr = (f-kc*kf)*Bsq;
    
	f = make_real3(vel.x.y, vel.y.y, vel.z.y);    
	kf = dot(kc,f)*invk2;
	vi = (f-kc*kf)*Bsq;

	/*Add the random velocities to global memory*/
	/*Velocities are stored as a complex number in each dimension, 
	  packed in a cufftComplex3 as 6 real numbers.
	  i.e three complex numbers one after the other*/
	kk = gridVelsFourier[icell_conj];
	gridVelsFourier[icell_conj] = {kk.x.x+vr.x, kk.x.y+vi.x,
				       kk.y.x+vr.y, kk.y.y+vi.y,
				       kk.z.x+vr.z, kk.z.y+vi.z};        
    
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
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(d_gridVels,
						make_real3(0), ncells);
      auto d_gridVelsFourier = thrust::raw_pointer_cast(gridVelsFourier.data());
      int ncellsF = ncells*3;
      Nthreads = BLOCKSIZE<ncellsF?BLOCKSIZE:ncellsF;
      Nblocks  =  ncellsF/Nthreads +  ((ncellsF%Nthreads!=0)?1:0); 
      fillWithGPU<<<Nblocks, Nthreads, 0, st>>>(d_gridVelsFourier,
						cufftComplex(), 3*ncells);
      

      Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0); 

      /*Gaussian spreading/interpolation kernel parameters, s(r) = prefactor*exp(-tau*r2)
	See eq. 13 in [2]
      */
      real3 prefactor = pow(2.0*psi*psi/(M_PI), 1.5)/make_real3(pow(eta.x,1.5), pow(eta.y, 1.5), pow(eta.z, 1.5));
      real3 tau       = 2.0*psi*psi/eta;
      sys->log<System::DEBUG2>("[BDHI::PSE] Particles to grid");
      /*Spread force on particles to grid positions -> S·F*/
      PSE_ns::particles2GridD<<<Nblocks*TPP, Nthreads, 0, st>>>
	(pos.raw(), v, d_gridVels, numberParticles, P, grid, prefactor, tau);


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
      /*Scale the wave space grid forces, transforming in velocities -> B·FFTf·S·F*/
      dim3 NthreadsCells = 128;
      dim3 NblocksCells;
      NblocksCells. x=  grid.cellDim.x/NthreadsCells.x + 1;
      NblocksCells. y=  grid.cellDim.y/NthreadsCells.y + 1;
      NblocksCells. z=  grid.cellDim.z/NthreadsCells.z + 1;
      auto d_fourierFactor = thrust::raw_pointer_cast(fourierFactor.data());
      PSE_ns::forceFourier2Vel<<<NblocksCells, NthreadsCells, 0, st>>>
	((PSE_ns::cufftComplex3*) d_gridVelsFourier,
	 (PSE_ns::cufftComplex3*) d_gridVelsFourier,
	 d_fourierFactor,
	 grid);

      /*The stochastic part only needs to be computed with T>0*/
      if(temperature > real(0.0)){
	sys->log<System::DEBUG2>("[BDHI::PSE] Wave space brownian noise");
	NblocksCells.x = NblocksCells.x/2+1;
	auto d_farNoise = thrust::raw_pointer_cast(farNoise.data());
	real prefactor = sqrt(2*temperature/dt/(grid.cellSize.x*grid.cellSize.y*grid.cellSize.z));
	/*Add the stochastic noise to the fourier velocities -> B·FFT·S·F + 1/√σ·√B·dWw*/
	PSE_ns::fourierBrownianNoise<<<NblocksCells, NthreadsCells, 0, st>>>(
									     d_farNoise,
									     (PSE_ns::cufftComplex3*)d_gridVelsFourier,
									     d_fourierFactor,
									     grid, prefactor);
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
	σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
      PSE_ns::grid2ParticlesD<<<Nblocks, Nthreads, 0, st>>>
	(pos.raw(), Mv, d_gridVels,
	 numberParticles, P, grid, prefactor, tau);
      sys->log<System::DEBUG2>("[BDHI::PSE] MF wave space Done");
      
    }



    void PSE::computeMF(real3* MF,     cudaStream_t st){
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
	  cl->transverseList(Mv_tr, st);
    
	}
      };
    }

    void PSE::computeBdW(real3* BdW, cudaStream_t st){
      sys->log<System::DEBUG2>("[BDHI::PSE] Real space brownian noise");
      /*Far contribution is in Mdot_far*/
      /*Compute stochastic term only if T>0 */
      if(temperature == real(0.0)) return;
      int numberParticles = pg->getNumberParticles();
      /****Near part*****/
      /*List transverser for near dot product*/
      PSE_ns::RPYNearTransverser<real3> tr(nullptr, nullptr,
					   *RPY_near,
					   M0, rcut, box);
      /*Functor for dot product*/
      PSE_ns::Dotctor Mvdot_near(tr, cl, numberParticles, st);
      /*Lanczos algorithm to compute M_near^1/2 · noise. See LanczosAlgorithm.cuh*/
      real *noise = lanczos->getV(numberParticles);
      curandGenerateNormal(curng, noise,
			   3*numberParticles + (3*numberParticles)%2,
			   real(0.0), real(1.0));
      lanczos->solve(Mvdot_near, (real *)BdW, noise, numberParticles, lanczosTolerance, st);       
    }

    void PSE::computeDivM(real3* divM, cudaStream_t st){}


    void PSE::finish_step(cudaStream_t st){
      sys->log<System::DEBUG2>("[BDHI::PSE] Finishing step");
 
    }
  }
#undef TPP
}

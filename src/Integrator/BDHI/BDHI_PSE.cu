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
#include<fstream>
#include"GPUutils.cuh"
#include<vector>
#include<algorithm>

using namespace BDHI;
using namespace std;



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
					   NeighbourList::Utils utils,  /*Grid information*/
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
    int icell = utils.getCellIndex(cell);
    if(cell.x>=utils.cellDim.x) return;
    if(cell.y>=utils.cellDim.y) return;
    if(cell.z>=utils.cellDim.z) return;

    /*K=0 doesnt contribute*/
    if(icell == 0){
      Bfactor[0] = make_real3(0.0);
      return;
    }
    /*The factors are computed in double precision for improved accuracy*/
    double3 pi2invL = double(M_PI)/make_double3(utils.Lhalf);

    /*Get my wave number*/
    double3 K = make_double3(cell)*pi2invL;
    /*Remember that FFT stores wave numbers as K=0:N/2+1:-N/2:-1 */    
    if(cell.x >= (utils.cellDim.x+1)/2) K.x -= double(utils.cellDim.x)*pi2invL.x;
    if(cell.y >= (utils.cellDim.y+1)/2) K.y -= double(utils.cellDim.y)*pi2invL.y;
    if(cell.z >= (utils.cellDim.z+1)/2) K.z -= double(utils.cellDim.z)*pi2invL.z;

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
    double3 hashimoto = (1.0 + k2_invpsi2_4)*make_double3(expf(tau.x), expf(tau.y), expf(tau.z))/k2;

    /*eq. 20.5 in [1]*/    
    double3 B = sink*sink*invk2*hashimoto/(vis*rh*rh);
    B /= double(utils.cellDim.x*utils.cellDim.y*utils.cellDim.z);
    /*Store theresult in global memory*/
    Bfactor[icell] = make_real3(B);    

  }

}
/*Constructor*/
PSE::PSE(real vis, /*viscosity*/
	 real T, /*Temperature*/
	 real rh,/*Hydrodynamic radius*/
	 real psi,/*RPY splitting parameter*/
	 int N, /*Number of particles*/
	 int max_iter /*Maximum number of Lanczos iterations for the near part*/
	 ):
  BDHI_Method(1/(6*M_PI*rh*vis), rh, N), T(T),
  psi(psi),
  lanczos(N, 0.05){
  cerr<<"\tInitializing PSE subsystem..."<<endl;
  /*Get the box size*/
  real3 L = gcnf.L;
  
  /*Kernel launching parameters*/
  BLOCKSIZE = 128;
  Nthreads = BLOCKSIZE<N?BLOCKSIZE:N;
  Nblocks  =  N/Nthreads +  ((N%Nthreads!=0)?1:0); 
  
  /* M = Mr + Mw */

  /*Compute M0*/
  double pi = M_PI;
  double a = rh;
  double prefac = (1.0/(24*sqrt(pi*pi*pi)*psi*a*a*vis));
  /*M0 = Mr(0) = F(0)(I-r^r) + G(0)(r^r) = F(0) = Mii_r . 
    See eq. 14 in [1] and RPYPSE_nearTextures*/
  this->M0 = prefac*(1-exp(-4*a*a*psi*psi)+4*sqrt(pi)*a*psi*std::erfc(2*a*psi));  

  /****Initialize near space part: Mr *******/
  real er = 1e-3; /*Short range error tolerance*/
  /*Near neighbour list cutoff distance, see sec II:C in [1]*/
  rcut = a*sqrt(-log(er))/psi;

  /*Initialize the neighbour list */
  this->cl = CellList(rcut);
  /*Initialize the near RPY textures*/
  nearTexs = RPYPSE_nearTextures(vis, rh, psi, M0, rcut, 4096);
  /*Initialize the Lanczos algorithm*/
  lanczos.init();

  /****Initialize wave space part: Mw ******/
  real ew = 1e-3; /*Long range error tolerance*/
  /*Maximum wave number for the far calculation*/
  kcut = 2*psi*sqrt(-log(ew))/a;
  /*Corresponding real space grid size*/
  double hgrid = 2*pi/kcut;

  /*Create a grid with cellDim cells*/
  /*This object will contain useful information about the grid,
    mainly number of cells, cell size and usual parameters for
    using it in the gpu, as invCellSize*/  
  int3 cellDim = make_int3(L/hgrid)+1;

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
  mesh = Mesh(cellDim, L, L.x/double(cellDim.x));

  /*Print information*/
  cerr<<"\t\tSplitting factor: "<<psi<<endl;  
  cerr<<"\t\tClose range Neighbour Cell List: "<<endl;
  cl.print();
  cerr<<"\t\tFar range Neighbour Cell List: "<<endl;
  mesh.print();

  
  cudaStreamCreate(&stream);
  cudaStreamCreate(&stream2);

  
  /*The quantity spreaded to the grid in real space*/
  /*The layout of this array is
    fx000, fy000, fz000, fx001, fy001, fz001..., fxnnn, fynnn, fznnn. n=ncells-1*/
  /*Can be Force when spreading particles to the grid and
    velocities when interpolating from the grid to the particles*/
  gridVels = Vector3(mesh.ncells);
  /*Same but in wave space, in MF it will be the forces that get transformed to velocities*/
  /*3 complex numbers per cell*/
  gridVelsFourier = Vector<cufftComplex>(3*mesh.ncells);

  /*Initialize far stochastic random generators*/
  if(T>real(0.0)){
    cerr<<"\t\tInitializing cuRand...."<<endl;
    int fnsize = mesh.ncells;
    fnsize += fnsize%2;
    farNoise = GPUVector<curandState>(fnsize);
    PSE_ns::initCurand<<<fnsize/32+1, 32,0, stream>>>(farNoise, gcnf.seed, fnsize);
  }
  /*Grid spreading/interpolation parameters*/
  /*Gaussian spreading/interpolation kernel support points neighbour distance
    See sec. 2.1 in [2]*/
  this->P = make_int3(1);
  double3 pw = make_double3(2*P+1);/*Number of support points*/
  double3 h = make_double3(mesh.utils.cellSize); /*Cell size*/
  /*Number of standard deviations in the grid's Gaussian kernel support*/
  double3 m = 0.976*sqrt(M_PI)*make_double3(sqrt(pw.x), sqrt(pw.y), sqrt(pw.z));
  /*Standard deviation of the Gaussian kernel*/
  double3 w   = pw*h/2.0;
  /*Gaussian splitting parameter*/
  this->eta = make_real3(pow(2.0*psi, 2)*w*w/(m*m));

  /*B in [1], this array stores, for each cell/fourier node,
    the scaling factor to go from forces to velocities in fourier space*/
  fourierFactor = GPUVector<real3>(mesh.ncells);
  /*Launch a thread per cell/node*/
  dim3 NthreadsCells = BLOCKSIZE;
  dim3 NblocksCells;
  NblocksCells. x=  mesh.cellDim.x/NthreadsCells.x + 1;
  NblocksCells. y=  mesh.cellDim.y/NthreadsCells.y + 1;
  NblocksCells. z=  mesh.cellDim.z/NthreadsCells.z + 1;

  PSE_ns::fillFourierScalingFactor<<<NblocksCells, NthreadsCells, 0, stream2>>>
    (fourierFactor.d_m, mesh.utils,
     rh,vis, psi, eta);
  
    
  /*Be sure everything starts at zero*/
  gridVels.memset(0);
  gridVelsFourier.memset(0);

  /*Set up cuFFT*/

  /*I will be handling workspace memory*/
  cufftSetAutoAllocation(cufft_plan_forward, 0);
  cufftSetAutoAllocation(cufft_plan_inverse, 0);

  int3 cdtmp = {mesh.cellDim.z, mesh.cellDim.y, mesh.cellDim.x};
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
    cerr<<"ERROR!: Setting up cuFFT Forward in BDHI_PSE!"<<endl; exit(1);}  
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
    cerr<<"ERROR!: Setting up cuFFT Inverse in BDHI_PSE!"<<endl; exit(1);}

  /*Allocate cuFFT work area*/
  size_t cufftWorkSizef = 0, cufftWorkSizei = 0;

  cufftGetSize(cufft_plan_forward, &cufftWorkSizef);
  cufftGetSize(cufft_plan_inverse, &cufftWorkSizei);

  size_t cufftWorkSize = std::max(cufftWorkSizef, cufftWorkSizei);

  cerr<<"\t\tNecessary work space for cuFFT: "<<printUtils::prettySize(cufftWorkSize)<<endl;
  size_t free_mem, total_mem;
  cuMemGetInfo(&free_mem, &total_mem);

  if(free_mem<cufftWorkSize){
    cerr<<"\t\tERROR: Not enough memory in device to allocate cuFFT!!, try lowering the splitting parameter!"<<endl;
    exit(1);    
  }

  cufftWorkArea = GPUVector<real>(cufftWorkSize/sizeof(real)+1);
  
  cufftSetWorkArea(cufft_plan_forward, (void*)cufftWorkArea.d_m);
  cufftSetWorkArea(cufft_plan_inverse, (void*)cufftWorkArea.d_m);
  
  cudaDeviceSynchronize();
  cerr<<"DONE!!"<<endl;  
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

  /*Ensure the result array is set to zero*/
  cudaMemset((real*)Mv, 0, N*sizeof(real3));

  // Mdot_nearThread = std::thread(&PSE::Mdot_near<vtype>, this, Mv, v, st);
  // Mdot_nearThread.join();  
  
  Mdot_near<vtype>(Mv, v, st);
  Mdot_farThread = std::thread(&PSE::Mdot_far<vtype>, this, Mv, v, stream2);
  //Mdot_far<vtype>(Mv, v, st);
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
		       cudaTextureObject_t texF,
		       cudaTextureObject_t texG,		       
		       real M0, /*RPY_near(0)*/
		       real rcut,/*cutoff distance*/
		       BoxUtils box/*Contains information and methods about the box, like apply_pbc*/
		       ):
      v(v), Mv(Mv), texF(texF), texG(texG), M0(M0), rcut(rcut), box(box){
      invrcut2 = 1/(rcut*rcut);
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
      real3 rij = make_real3(pj)-make_real3(pi);
      box.apply_pbc(rij);
      
      const real r2 = dot(rij, rij);      
      const float r2c = r2*invrcut2;      
      /*Predication seems to work geat here, so the checkings are done as soon as possible*/
      /*If i==j */
      if(r2==real(0.0)){
	/*M_ii·vi = M0*I·vi */
	return vj;
      }
      /*Many particles fall outside rcut, so checking here improves eficiency*/
      if(r2c>=1.0f) return make_real3(0);


      
      /*M0 := Mii := RPY_near(0) is multiplied once at the end*/
      /*Fetch RPY coefficients, see RPYPSE_nearTextures*/
      /* Mreal(r) = M0*(F(r)·I + (G(r)-F(r))·rr) */
      const real f = (real)tex1D<float>(texF, r2c);
      const real g = (real)tex1D<float>(texG, r2c);

      /*Update the result with Mr_ij·vj, the current box dot the current three elements of v*/
      /*This expression is a little obfuscated, Mr_ij·vj*/
      /*
	Mr = f(r)*I+(g(r)-f(r))*r(diadic)r - > (M·v)_ß = f(r)·v_ß + (g(r)-f(r))·v·(r(diadic)r)
	Where f and g are the RPY coefficients
      */
      const real gmfv = (g-f)*dot(rij, vj)/r2;
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
    cudaTextureObject_t texF, texG;
    real rcut, invrcut2;
    BoxUtils box;
  };
}

/*Compute Mr·v*/
template<typename vtype>
void PSE::Mdot_near(real3 *Mv, vtype *v, cudaStream_t st){
  /*Near contribution*/
  BoxUtils box(gcnf.L);
  /*Helper space for sorted v, only if the neighbour list needs reordering*/
  static GPUVector<vtype> sortV;
  if(cl.needsReorder())
    if(sortV.size()!=N) sortV = GPUVector<vtype>(N);

  /*Create the Transverser struct*/  
  PSE_ns::RPYNearTransverser<vtype> tr(sortV.d_m, Mv,
				       nearTexs.getFtex(), nearTexs.getGtex(),
				       M0, rcut, box);

  /*Update the list if needed*/
  cl.makeNeighbourList(st);
  /*Reorder the input vector if needed*/
  cl.reorderProperty(v, sortV.d_m, N, st);
  cl.transverse(tr, st);

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
				  NeighbourList::Utils utils, /*Grid information and methods*/
				  BoxUtils box, /*Box information and methods*/
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
    int3 celli = utils.getCell(pi);
    
    int3 cellj;    
    int x,y,z;
    /*Conversion between cell number and cell center position*/
    real3 cellPosOffset = real(0.5)*utils.cellSize-utils.Lhalf;

    /*Transverse the Pth neighbour cells*/
    for(x=-P.x+offset; x<=P.x; x+=TPP) 
      for(z=-P.z; z<=P.z; z++)
	for(y=-P.y; y<=P.y; y++){
	  /*Get the other cell*/
	  cellj = celli+make_int3(x,y,z);
	  /*Corrected with PBC*/
	  utils.pbc_cell(cellj);

	  /*Get index of cell j*/
	  int jcell = utils.getCellIndex(cellj);
	  /*Distance from particle i to center of cell j*/
	  real3 rij = pi-make_real3(cellj)*utils.cellSize-cellPosOffset;
	  box.apply_pbc(rij);
	  
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
	      NeighbourList::Utils utils/*Grid information and methods*/
				   ){
    /*Get my cell*/
    int3 cell;
    cell.x= blockIdx.x*blockDim.x + threadIdx.x;
    cell.y= blockIdx.y*blockDim.y + threadIdx.y;
    cell.z= blockIdx.z*blockDim.z + threadIdx.z;
    
    if(cell.x>=utils.cellDim.x/2+1) return;
    if(cell.y>=utils.cellDim.y) return;
    if(cell.z>=utils.cellDim.z) return;
    int icell = utils.getCellIndex(cell);

    real3 pi2invL = real(M_PI)/utils.Lhalf;
    /*My wave number*/
    real3 k = make_real3(cell)*pi2invL;

    /*Be careful with the conjugates*/
    if(cell.x >= (utils.cellDim.x+1)/2) k.x -= real(utils.cellDim.x)*pi2invL.x;
    if(cell.y >= (utils.cellDim.y+1)/2) k.y -= real(utils.cellDim.y)*pi2invL.y;
    if(cell.z >= (utils.cellDim.z+1)/2) k.z -= real(utils.cellDim.z)*pi2invL.z;

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
				  NeighbourList::Utils utils, /*Grid information and methods*/
				  BoxUtils box, /*Box information and methods*/
				  real3 prefactor,/*Prefactor for the kernel, (2*xi*xi/(pi·eta))^3/2*/
				  real3 tau /*Kernel exponential factor, 2*xi*xi/eta*/
				  ){
    /*A thread per particle */
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id>=N) return;
    /*Get my particle and my cell*/
    
    const real3 pi = make_real3(pos[id]);
    const int3 celli = utils.getCell(pi);
    int3 cellj;
    /*The S^T = St = σ S*/    
    prefactor *= (utils.cellSize.x*utils.cellSize.y*utils.cellSize.z);

    real3  result = make_real3(0);
    
    int x,y,z;
    /*Transform cell number to cell center position*/
    real3 cellPosOffset = real(0.5)*utils.cellSize-utils.Lhalf;
    /*Transvers the Pth neighbour cells*/
    for(z=-P.z; z<=P.z; z++)
      for(y=-P.y; y<=P.y; y++)
	for(x=-P.x; x<=P.x; x++){
	  /*Get neighbour cell*/
	  cellj = celli+make_int3(x,y,z);
	  utils.pbc_cell(cellj);
	  int jcell = utils.getCellIndex(cellj);

	  /*Compute distance to center*/
	  real3 rij = pi-make_real3(cellj)*utils.cellSize - cellPosOffset;
	  box.apply_pbc(rij);
	  real r2 = dot(rij, rij);
	  /*Interpolate cell value and sum*/
	  real3 cellj_vel = gridVels[jcell];
	  result += prefactor*make_real3(expf(-tau.x*r2), expf(-tau.y*r2), expf(-tau.z*r2))*cellj_vel;
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
	  NeighbourList::Utils utils, /*Grid parameters. Size of a cell, number of cells...*/
	  real prefactor/* sqrt(2·T/dt) */
				       ){

    /*Get my cell*/
    int3 cell;
    cell.x= blockIdx.x*blockDim.x + threadIdx.x;
    cell.y= blockIdx.y*blockDim.y + threadIdx.y;
    cell.z= blockIdx.z*blockDim.z + threadIdx.z;

    /*Get my cell index*/
    const int icell = utils.getCellIndex(cell);

    /*cuFFT R2C and C2R only store half of the innermost dimension, the one that varies the fastest
      
      The input of R2C is real and the output of C2R is real. 
      The only way for this to be true is if v_k={i,j,k} = v*_k{N-i, N-j, N-k}

      So the conjugates are redundant and the is no need to compute them nor storage them.      
     */
    /*Only compute the first half of the innermost dimension!*/
    if(cell.x>=utils.cellDim.x/2+1) return;
    if(cell.y>=utils.cellDim.y) return;
    if(cell.z>=utils.cellDim.z) return;
    
    /*K=0 is not added, no stochastic motion is added to the center of mass*/
    if(icell == 0){ return;}
    /*Fetch my rng*/
    curandState_t *rng = &farNoise[icell];

    /*Corresponding conjugate wave number,
      as the innermost dimension conjugates are redundant, computing the conjugate is needed
      only when cell.x == 0
      
     Note that this line works even on nyquist points
     */
    const int3 cell_conj = {(cell.x?(utils.cellDim.x-cell.x):cell.x),
			    (cell.y?(utils.cellDim.y-cell.y):cell.y),
			    (cell.z?(utils.cellDim.z-cell.z):cell.z)};    

    
    const int icell_conj = utils.getCellIndex(cell_conj);
    
    /*Compute the wave number of my cell and its conjugate*/
    const real3 pi2invL = real(M_PI)/utils.Lhalf;
    
    real3 k = make_real3(cell)*pi2invL;
    real3 kc = make_real3(cell_conj)*pi2invL;
    
    // /*Which is mirrored beyond cell_a = Ncells_a/2 */
    if(cell.x >= (utils.cellDim.x+1)/2) k.x -= real(utils.cellDim.x)*pi2invL.x;
    if(cell.y >= (utils.cellDim.y+1)/2) k.y -= real(utils.cellDim.y)*pi2invL.y;
    if(cell.z >= (utils.cellDim.z+1)/2) k.z -= real(utils.cellDim.z)*pi2invL.z;

    if(cell_conj.x >= (utils.cellDim.x+1)/2) kc.x -= real(utils.cellDim.x)*pi2invL.x;
    if(cell_conj.y >= (utils.cellDim.y+1)/2) kc.y -= real(utils.cellDim.y)*pi2invL.y;
    if(cell_conj.z >= (utils.cellDim.z+1)/2) kc.z -= real(utils.cellDim.z)*pi2invL.z;
    

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
    bool isXnyquist = (cell.x == utils.cellDim.x/2) && (utils.cellDim.x/2 == (utils.cellDim.x+1)/2);
    bool isYnyquist = (cell.y == utils.cellDim.y/2) && (utils.cellDim.y/2 == (utils.cellDim.y+1)/2);
    bool isZnyquist = (cell.z == utils.cellDim.z/2) && (utils.cellDim.z/2 == (utils.cellDim.z+1)/2);

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

  cufftSetStream(cufft_plan_forward, st);
  cufftSetStream(cufft_plan_inverse, st);

  /*Clean gridVels*/
  cudaMemsetAsync(gridVels.d_m, 0, mesh.ncells*sizeof(real3), st);
  cudaMemsetAsync(gridVelsFourier.d_m, 0, 3*mesh.ncells*sizeof(cufftComplex), st);


  /*Gaussian spreading/interpolation kernel parameters, s(r) = prefactor*exp(-tau*r2)
    See eq. 13 in [2]
   */
  real3 prefactor = pow(2.0*psi*psi/(M_PI), 1.5)/make_real3(pow(eta.x,1.5), pow(eta.y, 1.5), pow(eta.z, 1.5));
  real3 tau       = 2.0*psi*psi/eta;
  
  /*Spread force on particles to grid positions -> S·F*/
  BoxUtils box(gcnf.L);
  PSE_ns::particles2GridD<<<Nblocks*TPP, Nthreads,0, st>>>
    (pos, v, gridVels.d_m, N, P, mesh.utils, box, prefactor, tau);



  /*Take the grid spreaded forces and apply take it to wave space -> FFTf·S·F*/
  auto cufftStatus =
    cufftExecR2C(cufft_plan_forward,
		 (cufftReal*)gridVels.d_m,
		 (cufftComplex*)gridVelsFourier.d_m);
  if(cufftStatus != CUFFT_SUCCESS){ cerr<<"Error in forward CUFFT "<<endl; exit(1);}


  /*Scale the wave space grid forces, transforming in velocities -> B·FFTf·S·F*/
  dim3 NthreadsCells = BLOCKSIZE;
  dim3 NblocksCells;
  NblocksCells. x=  mesh.cellDim.x/NthreadsCells.x + 1;
  NblocksCells. y=  mesh.cellDim.y/NthreadsCells.y + 1;
  NblocksCells. z=  mesh.cellDim.z/NthreadsCells.z + 1;
  
  PSE_ns::forceFourier2Vel<<<NblocksCells, NthreadsCells, 0, st>>>
           ((PSE_ns::cufftComplex3*) gridVelsFourier.d_m,
	    (PSE_ns::cufftComplex3*) gridVelsFourier.d_m,
	    fourierFactor.d_m,
	    mesh.utils);

  /*The stochastic part only needs to be computed with T>0*/
  if(T > real(0.0)){
    NblocksCells.x = NblocksCells.x/2+1;
    
    real prefactor = sqrt(2*T/gcnf.dt/(mesh.utils.cellSize.x*mesh.utils.cellSize.y*mesh.utils.cellSize.z));
    /*Add the stochastic noise to the fourier velocities -> B·FFT·S·F + 1/√σ·√B·dWw*/
    PSE_ns::fourierBrownianNoise<<<NblocksCells, NthreadsCells, 0, st>>>(
				       farNoise.d_m,
				       (PSE_ns::cufftComplex3*)gridVelsFourier.d_m,
				       fourierFactor.d_m,
				       mesh.utils, prefactor);
  }

  /*Take the fourier velocities back to real space ->  FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
  cufftStatus =
    cufftExecC2R(cufft_plan_inverse,
		 (cufftComplex*)gridVelsFourier.d_m,
		 (cufftReal*)gridVels.d_m);
  if(cufftStatus != CUFFT_SUCCESS){ cerr<<"Error in inverse CUFFT"<<endl; exit(1);}

  /*Interpolate the real space velocities back to the particle positions ->
    σ·St·FFTi·(B·FFT·S·F + 1/√σ·√B·dWw )*/
  PSE_ns::grid2ParticlesD<<<Nblocks, Nthreads, 0, st>>>
    (pos.d_m, Mv, gridVels.d_m,
     N, P, mesh.utils, box, prefactor, tau);  
}



void PSE::computeMF(real3* MF,     cudaStream_t st){
  Mdot<real4>(MF, force.d_m, st);
}



namespace PSE_ns{
  /*LanczosAlgorithm needs a functor that computes the product M·v*/
  /*Dotctor takes a list transverser and a cell list on construction, 
    and the operator () takes an array v and returns the product M·v*/
  struct Dotctor{
    /*Dotctor uses the same transverser as in Mr·F*/
    typedef typename PSE_ns::RPYNearTransverser<real3> myTransverser;
    myTransverser Mv_tr;
    CellList *cl;
    cudaStream_t st;
    Dotctor(myTransverser Mv_tr, CellList *cl, cudaStream_t st): Mv_tr(Mv_tr), cl(cl), st(st){ }

    inline void operator()(real3* Mv, real3 *v){
      /*Static storage for reordering the input vector*/
      static GPUVector3 sortV;
      if(cl->needsReorder())
	if(sortV.size() != gcnf.N) sortV = GPUVector3(gcnf.N);
      /*Clean the result array just in case*/
      cudaMemsetAsync(Mv, 0, gcnf.N*sizeof(real3), st);

      /*No need to remake the list each time, makeNeighbourList takes care of that*/
      //cl->makeNeighbourList(st);
      /*Reorder the input array if needed by the list*/
      cl->reorderProperty(v, sortV.d_m, sortV.size(), st);      

      /*Update the transverser input and output arrays*/
      Mv_tr.v = sortV.d_m;      
      Mv_tr.Mv = Mv;

      /*Perform the dot product*/
      cl->transverse(Mv_tr, st);
    
    }
  };
}

void PSE::computeBdW(real3* BdW, cudaStream_t st){
  /*Far contribution is in Mdot_far*/
  /*Compute stochastic term only if T>0 */
  if(T == real(0.0)) return;
  /****Near part*****/
  BoxUtils box(gcnf.L);
  /*List transverser for near dot product*/
  PSE_ns::RPYNearTransverser<real3> tr(nullptr, nullptr,
    nearTexs.getFtex(), nearTexs.getGtex(),
    M0, rcut, box);
  /*Functor for dot product*/
  PSE_ns::Dotctor Mvdot_near(tr, &cl, st);
  /*Lanczos algorithm to compute M_near^1/2 · noise. See LanczosAlgorithm.cuh*/

  lanczos.solveNoise(Mvdot_near, (real *)BdW, st);

  
}

void PSE::computeDivM(real3* divM, cudaStream_t st){}


void PSE::finish_step(cudaStream_t st){
  Mdot_farThread.join();
  cudaDeviceSynchronize();


}

RPYPSE_nearTextures::RPYPSE_nearTextures(real vis, real rh, real psi, real m0, real rcut, int ntab):
  vis(vis), rh(rh), psi(psi), FGPU(nullptr), GGPU(nullptr){
	  
  this->M0 = 6*M_PI*m0;
  float Fcpu[ntab], Gcpu[ntab];
  
  Fcpu[0] = 1.0f;
  Gcpu[0] = 1.0f;
  
  double rc2 = rcut*rcut;
  double dr2 = rc2/(double)ntab;
  double r2 = 0.5*dr2;
  //  ofstream out("FG2.dat");
  //out<<sqrt(r2-0.5*dr2)<<" "<<Fcpu[0]+(Gcpu[0]-Fcpu[0])<<endl;

  fori(1, ntab-1){
    r2 += dr2;
    double2 tmp = FandG(sqrt(r2));
    Fcpu[i] = tmp.x;
    Gcpu[i] = tmp.y;
    //out<<sqrt(r2-0.5*dr2)<<" "<<Fcpu[i]+(Gcpu[i]-Fcpu[i])<<endl;

  }
  
  Fcpu[ntab-1] = 0.0f;
  Gcpu[ntab-1] = 0.0f;
  //out<<sqrt(r2+dr2-0.5*dr2)<<" "<<Fcpu[ntab-1]+(Gcpu[ntab-1]-Fcpu[ntab-1])<<endl;
  

    cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc(32, 0,0,0, cudaChannelFormatKindFloat);
    gpuErrchk(cudaMallocArray(&FGPU,
			      &channelDesc,
			      ntab,1));
    gpuErrchk(cudaMallocArray(&GGPU,
			      &channelDesc,
			      ntab,1));
  
    gpuErrchk(cudaMemcpyToArray(FGPU, 0,0, Fcpu, ntab*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToArray(GGPU, 0,0, Gcpu, ntab*sizeof(float), cudaMemcpyHostToDevice));



    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.normalizedCoords = 1;

    resDesc.res.array.array = FGPU;
    gpuErrchk(cudaCreateTextureObject(&texF, &resDesc, &texDesc, NULL));
    resDesc.res.array.array = GGPU;
    gpuErrchk(cudaCreateTextureObject(&texG, &resDesc, &texDesc, NULL));
  
}

double2 RPYPSE_nearTextures::FandG(double r){

  double a2mr = 2*rh-r;
  double a2pr = 2*rh+r;


  double rh2 = rh*rh;
  double rh4 = rh2*rh2;

  double psi2 = psi*psi;
  double psi3 = psi2*psi;
  double psi4 = psi2*psi2;

  double r2 = r*r;
  double r3 = r2*r;
  double r4 = r3*r;
      
  double f0, f1, f2, f3, f4 ,f5, f6, f7;
  double g0, g1, g2, g3, g4 ,g5, g6, g7;

      
  if(r>2*rh){
    f0 =(64.0*rh4*psi4 + 96.0*rh2*r2*psi4
	 - 128.0*rh*r3*psi4 + 36.0*r4*psi4-3.0)/
      (128.0*rh*r3*psi4);

    f4 = (3.0-4.0*psi4*a2mr*a2mr*(4.0*rh2+4.0*rh*r+9.0*r2))/
      (256.0*rh*r3*psi4);

    f5 = 0;


    g0 = (-64.0*rh4*psi4+96.0*rh2*r2*psi4-64.0*rh*r3*psi4 + 12.0*r4*psi4 +3.0)/
      (64.0*rh*r3*psi4);
	
	
    g4 = (4.0*psi4*a2mr*a2mr*a2mr*(2.0*rh+3.0*r)-3.0)/(128.0*rh*r3*psi4);

    g5 = 0;

      
  }
  else{
    f0 = (-16.0*rh4-24.0*rh2*r2+32.0*rh*r3-9.0*r4)/
      (32.0*rh*r3);

    f4 = 0;

    f5 = (4.0*psi4*a2mr*a2mr*(4.0*rh2+4.0*rh*r+9.0*r2)-3.0)/
      (256.0*rh*r3*psi4);


    g0 = a2mr*a2mr*a2mr*(2.0*rh+3.0*r)/(16.0*rh*r3);

    g4 = 0;

    g5 = (3.0 - 4.0*psi4*a2mr*a2mr*a2mr*(2.0*rh+3.0*r))/(128.0*rh*r3*psi4);
	
  }
  f1 = (-2.0*psi2*a2pr*(4.0*rh2-4.0*rh*r+9.0*r2) + 2.0*rh -3.0*r)/
    (128.0*rh*r3*psi3*sqrt(M_PI));

  f2 = (2.0*psi2*a2mr*(4.0*rh2+4.0*rh*r+9.0*r2)-2.0*rh-3.0*r)/
    (128.0*rh*r3*psi3*sqrt(M_PI));

  f3 = 3.0*(6.0*r2*psi2+1.0)/(64.0*sqrt(M_PI)*rh*r2*psi3);
      
  f6 = (4.0*psi4*a2pr*a2pr*(4.0*rh2-4.0*rh*r+9.0*r2)-3.0)/
    (256.0*rh*r3*psi4);

  f7 = 3.0*(1.0-12.0*r4*psi4)/(128.0*rh*r3*psi4);

  g1 = (2.0*psi2*a2pr*a2pr*(2.0*rh-3.0*r)-2.0*rh+3.0*r)/
    (64.0*sqrt(M_PI)*rh*r3*psi3);

  g2 = (-2.0*psi2*a2mr*a2mr*(2.0*rh+3.0*r)+2.0*rh+3.0*r)/
    (64.0*sqrt(M_PI)*rh*r3*psi3);

  g3 = (3.0*(2.0*r2*psi2-1.0))/(32.0*sqrt(M_PI)*rh*r2*psi3);

  g6 = (3.0-4.0*psi4*(2.0*rh-3.0*r)*a2pr*a2pr*a2pr)/(128.0*rh*r3*psi4);
      
  g7 = -3.0*(4.0*r4*psi4+1.0)/(64.0*rh*r3*psi4);
        
  return {params2FG(r, f0, f1, f2, f3, f4, f5, f6, f7)/(vis*rh*M0),
      params2FG(r, g0, g1, g2, g3, g4, g5, g6, g7)/(vis*rh*M0)};
}



double RPYPSE_nearTextures::params2FG(double r,
				      double f0, double f1, double f2, double f3,
				      double f4, double f5, double f6, double f7){
  double psisq = psi*psi;
  double a2mr = 2*rh-r;
  double a2pr = 2*rh+r;
  double rsq = r*r;      
  return  f0 + f1*exp(-psisq*a2pr*a2pr)  +
     f2*exp(-a2mr*a2mr*psisq) + f3*exp(-psisq*rsq) +
     f4*erfc(a2mr*psi) + f5*erfc(-a2mr*psi) +
     f6*erfc(a2pr*psi) + f7*erfc(r*psi);

}



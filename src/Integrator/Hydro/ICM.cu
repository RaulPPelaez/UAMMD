/*Raul P. Pelaez 2018-2021. Inertiam Coupling Method for particles in an incompressible fluctuating fluid.

See ICM.cuh for more info.

REFERENCES:

[1] Inertial coupling method for particles in an incompressible fluctuating fluid. Florencio Balboa, Rafael Delgado-Buscalioni, Boyce E. Griffith and Aleksandar Donev. 2014 https://doi.org/10.1016/j.cma.2013.10.029.
[2] Staggered Schemes for Fluctuating Hydrodynamics. Florencio Balboa, et.al. 2012.
[3] Brownian Dynamics without Green's Functions. Steve Delong, Florencio Balboa, et. al. 2014

 */
#include"ICM.cuh"
#include"third_party/saruprng.cuh"
#include"utils/cufftDebug.h"
#include"utils/curandDebug.h"
#include"utils/atomics.cuh"
#include"utils/debugTools.h"
#include"utils/cuda_lib_defines.h"
#include<iostream>
namespace uammd{
  namespace Hydro{
    namespace ICM_ns{
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
      //Computes S·F and adds it to gridVels
      template<class Kernel = IBM_kernels::Peskin::threePoint>
      __global__ void spreadParticleForces(real4 *pos,
					   real4 *force,
					   real3* gridData,
					   Grid grid,
					   int numberParticles,
					   real prefactor,
					   Kernel kernel){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=numberParticles) return;
	const real3 pi = make_real3(pos[id]);
	const real3 spreadQuantity = make_real3(force[id])*prefactor;
	//Corresponding cell of each direction in the staggered grid
	const real3 h = grid.cellSize;
	const int3 cellix = grid.getCell(make_real3(pi.x - real(0.5)*h.x, pi.y, pi.z));
	const int3 celliy = grid.getCell(make_real3(pi.x, pi.y - real(0.5)*h.y, pi.z));
	const int3 celliz = grid.getCell(make_real3(pi.x, pi.y, pi.z - real(0.5)*h.z));
	const real3 cellPosOffset = real(0.5)*(h - grid.box.boxSize); //Cell index to cell center position
	constexpr int P = Kernel::support/2;
	constexpr int supportCells = Kernel::support;
	constexpr int numberNeighbourCells = supportCells*supportCells*supportCells;
	for(int i = 0; i<numberNeighbourCells; i++){
	  //Contribution to vx
	  {
	    //Compute neighbour cell index
	    int3 celljx = make_int3(cellix.x + i%supportCells - P,
				    cellix.y + (i/supportCells)%supportCells - P,
				    cellix.z + i/(supportCells*supportCells) - P );
	    celljx = grid.pbc_cell(celljx);

	    const int jcellx = grid.getCellIndex(celljx);
	    /*Staggered distance from q - noise to center of cell j*/
	    const real3 rijx = pi-make_real3(celljx)*h-cellPosOffset;
	    //Spread Wx
	    auto r = grid.box.apply_pbc({rijx.x - real(0.5)*h.x, rijx.y, rijx.z});
	    real fx = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*spreadQuantity.x;
	    atomicAdd(&gridData[jcellx].x, fx);
	  }
	  //Contribution to vy
	  {
	    int3 celljy = make_int3(celliy.x + i%supportCells - P,
				    celliy.y + (i/supportCells)%supportCells - P,
				    celliy.z + i/(supportCells*supportCells) - P );
	    celljy = grid.pbc_cell(celljy);
	    const int jcelly = grid.getCellIndex(celljy);
	    const real3 rijy = pi - make_real3(celljy)*h-cellPosOffset;
	    auto r = grid.box.apply_pbc({rijy.x, rijy.y - real(0.5)*h.y, rijy.z});
	    real fy = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*spreadQuantity.y;
	    atomicAdd(&gridData[jcelly].y, fy);
	  }
	  //Contribution to vz
	  {
	    int3 celljz = make_int3(celliz.x + i%supportCells - P,
				    celliz.y + (i/supportCells)%supportCells - P,
				    celliz.z + i/(supportCells*supportCells) - P );
	    celljz = grid.pbc_cell(celljz);
	    const int jcellz = grid.getCellIndex(celljz);
	    const real3 rijz = pi-make_real3(celljz)*h-cellPosOffset;
	    auto r = grid.box.apply_pbc({rijz.x, rijz.y, rijz.z - real(0.5)*h.z});
	    real fz = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*spreadQuantity.z;
	    atomicAdd(&gridData[jcellz].z, fz);
	  }
	}

      }

      using cufftComplex3 = ICM::cufftComplex3;

      //Computes thermal drift term using RFD
      //kbT/\delta [ S(q^n + \delta/2\hat{W}^n) - S(q^n - \delta/2\hat{W}^n) ] ·\hat{W}^n
      //See eq. 32 and 33 in [3]
      template<class Kernel>
      __global__ void addThermalDrift(real4 *pos,
				      real3* gridVels,
				      Grid grid,
				      real driftPrefactor, //kbT/deltaRDF
				      real deltaRFD,
				      int numberParticles,
				      Kernel kernel,
				      uint seed, uint step){
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
	  W = make_real3(rng.gf(real(0.0), real(1.0)), rng.gf(real(0.0), real(1.0)).x);
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
	      auto r = grid.box.apply_pbc({rijx.x - real(0.5)*grid.cellSize.x, rijx.y, rijx.z});
	      SmSdWx = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.x;
	    }
	    {
	      /*Staggered distance from q - noise to center of cell j*/
	      const real3 rijx = q_m_noise-make_real3(celljx)*grid.cellSize-cellPosOffset;
	      //Spread Wx
	      auto r = grid.box.apply_pbc({rijx.x - real(0.5)*grid.cellSize.x, rijx.y, rijx.z});
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
	      auto r = grid.box.apply_pbc({rijy.x, rijy.y - real(0.5)*grid.cellSize.y, rijy.z});
	      SmSdWy = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.y;
	    }
	    {
	      const real3 rijy = q_m_noise-make_real3(celljy)*grid.cellSize-cellPosOffset;
	      auto r = grid.box.apply_pbc({rijy.x, rijy.y - real(0.5)*grid.cellSize.y, rijy.z});
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
	      auto r = grid.box.apply_pbc({rijz.x, rijz.y, rijz.z - real(0.5)*grid.cellSize.z});
	      SmSdWz = kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.z;
	    }
	    {
	      const real3 rijz = q_m_noise-make_real3(celljz)*grid.cellSize-cellPosOffset;
	      auto r = grid.box.apply_pbc({rijz.x, rijz.y, rijz.z - real(0.5)*grid.cellSize.z});
	      SmSdWz -= kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*W.z;
	    }
	    atomicAdd(&gridVels[jcellz].z, SmSdWz*driftPrefactor);
	  }
	}
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
	ICM::cufftComplex tmp;

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

      //Solves the velocity in eq. 36 in [1] given the Fourier representation of the RHS and stores it in gridVelsPrediction.
      //For that it applies the operator \mathcal{L}^-1 to the fluid forcing \vec{g}. See ICM.cuh.
      //If projectOnly is true, this kernel applies only the P operator, without the (\rho/dt I - \eta/2 L)^{-1}
      template<bool projectOnly = false>
      __global__ void solveStokesFourier(const cufftComplex3* fluidForcing, //input
					 cufftComplex3* gridVels,   //output
					 real viscosity,
					 real density,
					 real dt,
					 Grid grid, bool removeTotalMomemtum){
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
	if(icell==0){
	  if(removeTotalMomemtum)
	    gridVels[0] = cufftComplex3();//{0,0 ,0,0 ,0,0};
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
	  //Apply (\rho/dt·I-\eta/2·L) operator and project into divergence free space.
	  real prefactor = real(1.0);
	  if(!projectOnly){
	    const real L = -dot(keff, keff);
	    prefactor = real(1.0)/(real(1.0) - (dt/density)*real(0.5)*viscosity*L);
	  }
	  vk = projectFourier(keff, prefactor*vk);
	}
	const int ncells = grid.getNumberCells();
	//Store new velocity shifted back to cell faces, normalize FFT
	gridVels[icell] = (real(1.0)/real(ncells))*shiftVelocity(vk, cosk, sink);
      }

      //Midpoint step, takes particle positions to the next sub step.
      //In predictor mode takes q^n and replaces it by q^{n+1/2}, storing q^n in posOld
      //In corrector mode takes q^{n+1/2} and q^n and replaces pos by q^{n+1}
      enum class Step{PREDICTOR, CORRECTOR};
      template<Step mode = Step::PREDICTOR, class Kernel>
      __global__ void midPointStep(real4 *pos,
				   real4* posOld,
				   const real3* gridVels,
				   real dt,
				   Grid grid,
				   Kernel kernel,
				   int numberParticles){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= numberParticles) return;
	//Store q^n
	if(mode==Step::PREDICTOR){
	  posOld[id] = pos[id];
	}
	//q^n in predictor and q^{n+1/2} in corrector
	const real3 posCurrent = make_real3(pos[id]);
	//Staggered grid cells for interpolating each velocity coordinate
	const int3 cellix = grid.getCell(make_real3(posCurrent.x - real(0.5)*grid.cellSize.x, posCurrent.y, posCurrent.z));
	const int3 celliy = grid.getCell(make_real3(posCurrent.x, posCurrent.y - real(0.5)*grid.cellSize.y, posCurrent.z));
	const int3 celliz = grid.getCell(make_real3(posCurrent.x, posCurrent.y, posCurrent.z - real(0.5)*grid.cellSize.z));
	const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);
	constexpr int P = Kernel::support/2;
	constexpr int supportCells = Kernel::support;
	constexpr int numberNeighbourCells = supportCells*supportCells*supportCells;
	//J^T = dV·S
	const real dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
	//Pos prediction
	real3 pnew = make_real3(0);
	//Sum contribution of neighbouring cells
	for(int i = 0; i<numberNeighbourCells; i++){
	  //Apply J to the velocity of the current cell.
	  {
	    //Neighbour cell in the x direction
	    int3 celljx = make_int3(cellix.x + i%supportCells - P,
				    cellix.y + (i/supportCells)%supportCells - P,
				    cellix.z + i/(supportCells*supportCells) - P );
	    celljx = grid.pbc_cell(celljx);
	    //Cel lindex of neighbour
	    const int jcellx = grid.getCellIndex(celljx);
	    //Distance from particle i to center of cell j
	    const real3 rijx = posCurrent-make_real3(celljx)*grid.cellSize-cellPosOffset;
	    //p += J·v = dV·\delta(p_x_i-cell_x_j)·v_x_j
	    real v_jx = gridVels[jcellx].x;
	    auto r = grid.box.apply_pbc({rijx.x - real(0.5)*grid.cellSize.y, rijx.y, rijx.z});
	    pnew.x += kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*v_jx*dV;
	  }
	  {
	    int3 celljy = make_int3(celliy.x + i%supportCells - P,
				    celliy.y + (i/supportCells)%supportCells - P,
				    celliy.z + i/(supportCells*supportCells) - P );
	    celljy = grid.pbc_cell(celljy);
	    const int jcelly = grid.getCellIndex(celljy);
	    const real3 rijy = posCurrent-make_real3(celljy)*grid.cellSize-cellPosOffset;
	    real v_jy = gridVels[jcelly].y;
	    auto r = grid.box.apply_pbc({rijy.x, rijy.y - real(0.5)*grid.cellSize.y, rijy.z});
	    pnew.y += kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*v_jy*dV;
	  }
	  {
	    int3 celljz = make_int3(celliz.x + i%supportCells - P,
				    celliz.y + (i/supportCells)%supportCells - P,
				    celliz.z + i/(supportCells*supportCells) - P );
	    celljz = grid.pbc_cell(celljz);
	    const int jcellz = grid.getCellIndex(celljz);
	    const real3 rijz = posCurrent-make_real3(celljz)*grid.cellSize-cellPosOffset;
	    real v_jz = gridVels[jcellz].z;
	    auto r = grid.box.apply_pbc({rijz.x, rijz.y, rijz.z - real(0.5)*grid.cellSize.z});
	    pnew.z += kernel.phi(r.x)*kernel.phi(r.y)*kernel.phi(r.z)*v_jz*dV;
	  }
	}
	//Update position
	// p^{n+1/2} = p^n + dt/2 J^n v^n in predictor
	// p^{n+1} = p^n + dt J^{n+1/2}0.5(v^{n+1} + v^n) in corrector
	if(mode==Step::PREDICTOR){
	  pnew = posCurrent + real(0.5)*dt*pnew;
	}
	if(mode==Step::CORRECTOR){
	  pnew = make_real3(posOld[id]) + dt*pnew;
	}
	pos[id] = make_real4(pnew, pos[id].w);
      }

      struct Direction{
	static constexpr int XX = 0, YY=1, ZZ=2, XY=3, XZ=4, YZ=5;
      };

      //Computes the term sqrt(2·T·\eta/(dV dt)) D·W.
      __device__ real3 computeNoiseDivergence(int3 cell, int icell,
					      const real3* gridVels, //Real space velocities of each cell, defined at the cell faces
					      Grid grid,
					      real noisePrefactor,
					      //6 random numbers per cell, sorted by direction [first number for all cells, ..., second number for all cells,...]
					      const real* random){
	if(noisePrefactor == real(0.0)) return real3();
	const int ncells = grid.getNumberCells();
	//I will draw 6 random numbers for each cell
	real3 DW = make_real3(0);
	//(\nabla·W)_\alpha^i = \nabla^i·(W_{\alpha x}, W_{\alpha y}, W_{\alpha z}) ->
	// -> \partial_\alpha^i W_{\alpha\beta} = 1/d\alpha (W^{\alpha\beta}_{i+\alpha/2} - W^{\alpha\beta}_{i-\alpha/2})
	constexpr real sqrt2 = real(1.41421356237310);
	//Diagonal terms
	//See eq. 10 in [3] to understand the sqrt2
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
	return DW*noisePrefactor;
      }

      //Computes the Laplacian of the velocity: L·v^n
      __device__ real3 computeVelLaplacian(int3 cell, int icell,
					   const real3* gridVels,
					   Grid grid){
	real vx = gridVels[icell].x;
	real vy = gridVels[icell].y;
	real vz = gridVels[icell].z;
	int3 cellj;
	int icellj;
	real3 tmp;
	cellj = cell;
	cellj.x = grid.pbc_cell_coord<0>(cellj.x+1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_px = tmp.x;
	real vy_px = tmp.y;
	real vz_px = tmp.z;
	cellj = cell;
	cellj.x = grid.pbc_cell_coord<0>(cellj.x-1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_mx = tmp.x;
	real vy_mx = tmp.y;
	real vz_mx = tmp.z;
	cellj = cell;
	cellj.y = grid.pbc_cell_coord<1>(cellj.y+1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_py = tmp.x;
	real vy_py = tmp.y;
	real vz_py = tmp.z;
	cellj = cell;
	cellj.y = grid.pbc_cell_coord<1>(cellj.y-1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_my = tmp.x;
	real vy_my = tmp.y;
	real vz_my = tmp.z;
	cellj = cell;
	cellj.z = grid.pbc_cell_coord<2>(cellj.z+1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_pz = tmp.x;
	real vy_pz = tmp.y;
	real vz_pz = tmp.z;
	cellj = cell;
	cellj.z = grid.pbc_cell_coord<2>(cellj.z-1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_mz = tmp.x;
	real vy_mz = tmp.y;
	real vz_mz = tmp.z;
	real3 invh = grid.invCellSize;
	real3 velocityLaplacian;
	velocityLaplacian.x  = invh.x*invh.x * ( vx_px - real(2.0)*vx + vx_mx);
	velocityLaplacian.x += invh.y*invh.y * ( vx_py - real(2.0)*vx + vx_my);
	velocityLaplacian.x += invh.z*invh.z * ( vx_pz - real(2.0)*vx + vx_mz);
	velocityLaplacian.y  = invh.x*invh.x * ( vy_px - real(2.0)*vy + vy_mx);
	velocityLaplacian.y += invh.y*invh.y * ( vy_py - real(2.0)*vy + vy_my);
	velocityLaplacian.y += invh.z*invh.z * ( vy_pz - real(2.0)*vy + vy_mz);
	velocityLaplacian.z  = invh.x*invh.x * ( vz_px - real(2.0)*vz + vz_mx);
	velocityLaplacian.z += invh.y*invh.y * ( vz_py - real(2.0)*vz + vz_my);
	velocityLaplacian.z += invh.z*invh.z * ( vz_pz - real(2.0)*vz + vz_mz);
	return velocityLaplacian;
      }

      //Computes D·(\rho\vec{v}·\vec{v}^T)^n
      __device__ real3 computeAdvection(int3 cell, int icell,
					const real3* gridVels,
					Grid grid,
					real density){
	real vx = gridVels[icell].x;
	real vy = gridVels[icell].y;
	real vz = gridVels[icell].z;
	int3 cellj;
	int icellj;
	real3 tmp;
	cellj = cell;
	cellj.x = grid.pbc_cell_coord<0>(cellj.x+1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_px = tmp.x;
	real vy_px = tmp.y;
	real vz_px = tmp.z;
	cellj = cell;
	cellj.x = grid.pbc_cell_coord<0>(cellj.x-1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_mx = tmp.x;
	real vy_mx = tmp.y;
	real vz_mx = tmp.z;
	cellj = cell;
	cellj.y = grid.pbc_cell_coord<1>(cellj.y+1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_py = tmp.x;
	real vy_py = tmp.y;
	real vz_py = tmp.z;
	cellj = cell;
	cellj.y = grid.pbc_cell_coord<1>(cellj.y-1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_my = tmp.x;
	real vy_my = tmp.y;
	real vz_my = tmp.z;
	cellj = cell;
	cellj.z = grid.pbc_cell_coord<2>(cellj.z+1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_pz = tmp.x;
	real vy_pz = tmp.y;
	real vz_pz = tmp.z;
	cellj = cell;
	cellj.z = grid.pbc_cell_coord<2>(cellj.z-1);
	icellj = grid.getCellIndex(cellj);
	tmp = gridVels[icellj];
	real vx_mz = tmp.x;
	real vy_mz = tmp.y;
	real vz_mz = tmp.z;
	cellj = cell;
	cellj.x = grid.pbc_cell_coord<0>(cellj.x+1);
	cellj.y = grid.pbc_cell_coord<1>(cellj.y-1);
	icellj = grid.getCellIndex(cellj);
	real vy_px_my = gridVels[icellj].y;
	cellj = cell;
	cellj.x = grid.pbc_cell_coord<0>(cellj.x+1);
	cellj.z = grid.pbc_cell_coord<2>(cellj.z-1);
	icellj = grid.getCellIndex(cellj);
	real vz_px_mz = gridVels[icellj].z;
	cellj = cell;
	cellj.x = grid.pbc_cell_coord<0>(cellj.x-1);
	cellj.y = grid.pbc_cell_coord<1>(cellj.y+1);
	icellj = grid.getCellIndex(cellj);
	real vx_mx_py = gridVels[icellj].x;
	cellj = cell;
	cellj.z = grid.pbc_cell_coord<2>(cellj.z-1);
	cellj.y = grid.pbc_cell_coord<1>(cellj.y+1);
	icellj = grid.getCellIndex(cellj);
	real vz_py_mz = gridVels[icellj].z;
	cellj = cell;
	cellj.x = grid.pbc_cell_coord<0>(cellj.x-1);
	cellj.z = grid.pbc_cell_coord<2>(cellj.z+1);
	icellj = grid.getCellIndex(cellj);
	real vx_mx_pz = gridVels[icellj].x;
	cellj = cell;
	cellj.y = grid.pbc_cell_coord<1>(cellj.y-1);
	cellj.z = grid.pbc_cell_coord<2>(cellj.z+1);
	icellj = grid.getCellIndex(cellj);
	real vy_my_pz = gridVels[icellj].y;
	const real3 invh = grid.invCellSize;
	real3 advection;
	advection.x   = invh.x*( (vx_px + vx)*(vx_px + vx) - (vx + vx_mx)*(vx+ vx_mx) );
	advection.x  += invh.y*( (vx_py + vx)*(vy_px + vy) - (vx + vx_my)*(vy_px_my+ vy_my) );
	advection.x  += invh.z*( (vx_pz + vx)*(vz_px + vz) - (vx + vx_mz)*(vz_px_mz+ vz_mz) );
	advection.y   = invh.x*( (vy_px + vy)*(vx_py + vx) - (vy + vy_mx)*(vx_mx_py+ vx_mx) );
	advection.y  += invh.y*( (vy_py + vy)*(vy_py + vy) - (vy + vy_my)*(vy+ vy_my) );
	advection.y  += invh.z*( (vy_pz + vy)*(vz_py + vz) - (vy + vy_mz)*(vz_py_mz+ vz_mz) );
	advection.z   = invh.x*( (vz_px + vz)*(vx_pz + vx) - (vz + vz_mx)*(vx_mx_pz+ vx_mx) );
	advection.z  += invh.y*( (vz_py + vz)*(vy_pz + vy) - (vz + vz_my)*(vy_my_pz+ vy_my) );
	advection.z  += invh.z*( (vz_pz + vz)*(vz_pz + vz) - (vz + vz_mz)*(vz+ vz_mz) );
	advection *= real(0.25)*density;
	return advection;
      }

      //Any external force acting on the fluid
      __device__ real3 externalFluidForcing(int3 cell, int icell,
					    const real3* gridVels,
					    Grid grid){
	real3 f = real3();
	// real pos = (cell.x-real(0.5)*grid.cellDim.x + real(0.5))*grid.cellSize.x;
	// f.y = -sin(real(2.0*M_PI*2)*pos);
	return f;
      }

      //Computes fluid forcing \vec{g} (except the SF and thermal drift terms) for the unperturbed velocity field (unperturbed means that m_e = 0). See eq. 36 and 41 in [1]
      __global__ void updateCellVelocityUnperturbed(real3 *gridVels,
						    real3 *cellAdvection,
						    Grid grid,
						    real density,
						    real viscosity,
						    real noiseAmp,
						    real dt,
						    int ncells,
						    const real *random){
	//I expect a 3D grid of threads, one for each fourier node/grid cell
	//Get my cell
	int3 cell;
	cell.x= blockIdx.x*blockDim.x + threadIdx.x;
	cell.y= blockIdx.y*blockDim.y + threadIdx.y;
	cell.z= blockIdx.z*blockDim.z + threadIdx.z;
	//Get my cell index (position in the array)
	int icell =grid.getCellIndex(cell);
	if(cell.x>=grid.cellDim.x) return;
	if(cell.y>=grid.cellDim.y) return;
	if(cell.z>=grid.cellDim.z) return;
	real3 DivNoise = computeNoiseDivergence(cell, icell, gridVels, grid, noiseAmp, random);
	real3 Lv = computeVelLaplacian(cell, icell, gridVels, grid);
	real3 advection = computeAdvection(cell, icell, gridVels, grid, density);
	gridVels[icell] += (dt*viscosity*real(0.5)/density)*Lv +
	  DivNoise -
	  (dt/density)*(real(1.5)*advection - real(0.5)*cellAdvection[icell])
	  + dt/density*externalFluidForcing(cell, icell, gridVels, grid);
	cellAdvection[icell] = advection;
      }
    }


    ICM::ICM(shared_ptr<ParticleData> pd,
	     shared_ptr<System> sys,
	     Parameters par):
      Integrator(pd, sys, "Hydro::ICM"),
      dt(par.dt),
      temperature(par.temperature),
      density(par.density),
      viscosity(par.viscosity),
      box(par.box),
      removeTotalMomemtum(par.removeTotalMomemtum),
      sumThermalDrift(par.sumThermalDrift){
      sys->log<System::MESSAGE>("[Hydro::ICM] Initialized");
      CudaCheckError();
      seed = sys->rng().next32();
      int numberParticles = pg->getNumberParticles();
      if(density<0) sys->log<System::CRITICAL>("[Hydro::ICM] Please provide fluid density");
      if(viscosity<0)sys->log<System::CRITICAL>("[Hydro::ICM] Please provide fluid viscosity");
      if(par.hydrodynamicRadius>0 and par.cells.x > 0)
	sys->log<System::CRITICAL>("[Hydro::ICM] Please provide hydrodynamic radius OR cell dimensions, not both.");
      initializeGrid(par);
      double rh = getHydrodynamicRadius();
#ifndef SINGLE_PRECISION
      this->deltaRFD = 1e-6*rh;
#else
      this->deltaRFD = 1e-4*rh;
#endif
      printMessages(par);
      resizeContainers();
      initFluid();
      initCuFFT();
      initCuRAND();
      CudaSafeCall(cudaStreamCreate(&st));
      CudaCheckError();
    }

    ICM::~ICM(){
      cudaDeviceSynchronize();
      cudaStreamDestroy(st);
      (cufftDestroy(cufft_plan_inverse));
      (cufftDestroy(cufft_plan_forward));
      (curandDestroyGenerator(curng));
      sys->log<System::DEBUG>("[Hydro::ICM] Destroyed");
    }

    void ICM::initializeGrid(Parameters par){
      int3 cellDim = par.cells;
      //If cells were not provided compute the closest one for the requested Rh
      if(par.cells.x<0){
	if(par.hydrodynamicRadius<0)
	  sys->log<System::CRITICAL>("[BHDI::ICM] I need either the hydrodynamic radius or the number of cells!");
	real hgrid = par.hydrodynamicRadius/0.91;
	cellDim = make_int3(box.boxSize/hgrid);
	/*FFT likes a number of cells as cellDim.i = 2^n·3^l·5^m */
	cellDim = ICM_ns::nextFFTWiseSize3D(cellDim);
      }
      if(par.cells.x>0) cellDim = par.cells;
      if(cellDim.x <3)cellDim.x = 3;
      if(cellDim.y <3)cellDim.y = 3;
      if(cellDim.z==2)cellDim.z = 3;
      /*Store grid parameters in a Mesh object*/
      this->grid = Grid(box, cellDim);
    }

    void ICM::printMessages(Parameters par){
      /*Print information*/
      if(par.hydrodynamicRadius>0)
	sys->log<System::MESSAGE>("[Hydro::ICM] Target hydrodynamic radius: %f", par.hydrodynamicRadius);
      sys->log<System::MESSAGE>("[Hydro::ICM] Closest possible hydrodynamic radius: %f", getHydrodynamicRadius());
      sys->log<System::MESSAGE>("[Hydro::ICM] Box Size: %f %f %f", box.boxSize.x, box.boxSize.y, box.boxSize.z);
      sys->log<System::MESSAGE>("[Hydro::ICM] Mesh dimensions: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
      sys->log<System::MESSAGE>("[Hydro::ICM] Temperature: %f", temperature);
      sys->log<System::MESSAGE>("[Hydro::ICM] Fluid density: %f", density);
      sys->log<System::MESSAGE>("[Hydro::ICM] Fluid viscosity: %f", viscosity);
      sys->log<System::MESSAGE>("[Hydro::ICM] Self mobility: %f", this->getSelfMobility());
      sys->log<System::MESSAGE>("[Hydro::ICM] dt: %f", dt);
      if(sumThermalDrift and temperature > real(0.0)){
	sys->log<System::MESSAGE>("[Hydro::ICM] Random Finite Diference delta: %e", deltaRFD);
	sys->log<System::MESSAGE>("[Hydro::ICM] Summing thermal drift");
      }
      else
	sys->log<System::MESSAGE>("[Hydro::ICM] Not taking into account thermal drift");
      {
	real dx = grid.cellSize.x;
	real v = sqrt(temperature/(density*dx*dx*dx));
	real a = v*dt/dx;
	real b = viscosity*dt/(density*dx*dx);
	sys->log<System::MESSAGE>("[Hydro::ICM] CFL numbers: α = %f; β = %f", a, b);
	if(a > 1){
	  sys->log<System::WARNING>("[Hydro::ICM] CFL numbers above 1 can cause numerical issues and poor accuracy.");
	}
      }
    }

    void ICM::resizeContainers(){
      const int ncells = grid.getNumberCells();
      const int numberParticles = pg->getNumberParticles();
      try{
	cellAdvection.resize(ncells, real3());
	random.resize(6*ncells);
	posOld.resize(numberParticles);
	//gridVelsPrediction.resize(ncells);
	gridVelsPredictionF.resize(3*ncells);
	gridVels.resize(ncells, real3());
      }
      catch(thrust::system_error &e){
	sys->log<System::EXCEPTION>("[Hydro::ICM] Thrust could not allocate necessary arrays at initialization with error: %s", e.what());
	throw;
      }
    }

    void ICM::initCuFFT(){
      sys->log<System::DEBUG>("[Hydro::ICM] Initializing fluid.");
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
      sys->log<System::DEBUG>("[Hydro::ICM] cuFFT grid size: %d %d %d", cdtmp.x, cdtmp.y, cdtmp.z);
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
      sys->log<System::DEBUG>("[Hydro::ICM] Necessary work space for cuFFT: %s", printUtils::prettySize(cufftWorkSize).c_str());
      size_t free_mem, total_mem;
      CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
      if(free_mem<cufftWorkSize){
	sys->log<System::CRITICAL>("[Hydro::ICM] Not enough memory in device to allocate cuFFT free %s, needed: %s!!!",
				   printUtils::prettySize(free_mem).c_str(),
				   printUtils::prettySize(cufftWorkSize).c_str());
      }
      cufftWorkArea.resize(cufftWorkSize+1);
      void * d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      CufftSafeCall(cufftSetWorkArea(cufft_plan_forward, d_cufftWorkArea));
      CufftSafeCall(cufftSetWorkArea(cufft_plan_inverse, d_cufftWorkArea));
      CudaCheckError();
    }

    void ICM::initFluid(){
      sys->log<System::DEBUG>("[Hydro::ICM] Initializing fluid.");
      int ncells = grid.getNumberCells();
      thrust::host_vector<real3> cellVelocityCPU = gridVels;
      real cellVolume = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
      real3 * d_cellVelCPU = (real3*)thrust::raw_pointer_cast(cellVelocityCPU.data());
      fori(0, ncells){
	double fluidVelAmp = sqrt(temperature/(density*cellVolume));
	d_cellVelCPU[i] = make_real3(
				     sys->rng().gaussian(0.0, fluidVelAmp),
				     sys->rng().gaussian(0.0, fluidVelAmp),
				     sys->rng().gaussian(0.0, fluidVelAmp));
      }
      try{
	gridVels = cellVelocityCPU;
      }
      catch(thrust::system_error &e){
	sys->log<System::CRITICAL>("[Hydro::ICM] Thrust could not upload initial fluid velocity with error: %s", e.what());
      }
      //This is to initialize advection
      //unperturbedFluidForcing();
    }

    void ICM::initCuRAND(){
      sys->log<System::DEBUG>("[Hydro::ICM] Initializing cuRAND.");
      //Init rng
      CurandSafeCall(curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_DEFAULT));
      //curandCreateGenerator(&curng, CURAND_RNG_PSEUDO_MT19937);
      CurandSafeCall(curandSetPseudoRandomGeneratorSeed(curng, sys->rng().next()));
      thrust::device_vector<real> noise(30000);
      auto noise_ptr = thrust::raw_pointer_cast(noise.data());
      //Warm cuRNG
      CurandSafeCall(curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0));
      CurandSafeCall(curandGenerateNormal(curng, noise_ptr, noise.size(), 0.0, 1.0));
    }
    //Sum S·F term using the current particle positions
    void ICM::spreadParticleForces(){
      if(interactors.size()==0) return;
      int numberParticles = pg->getNumberParticles();
      int BLOCKSIZE = 128; /*threads per block*/
      int nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int nblocks = numberParticles/nthreads +  ((numberParticles%nthreads!=0)?1:0);
      auto indexIter = pg->getIndexIterator(access::location::gpu);
      {
	auto force = pd->getForce(access::location::gpu, access::mode::write);
	auto force_gr = pg->getPropertyIterator(force);
	thrust::fill(thrust::cuda::par, force_gr, force_gr + numberParticles, real4());
      }
      /*Compute new force*/
      for(auto forceComp: interactors) forceComp->sumForce(0);
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      const real prefactor = dt/density;
      ICM_ns::spreadParticleForces<<<nblocks, nthreads, 0, st>>>(pos.raw(),
								 force.raw(),
								 (real3*)d_gridVels,
								 grid,
								 numberParticles,
								 prefactor,
								 Kernel(grid.cellSize.x));
      CudaCheckError();
    }
    //Sum thermal drift term using random Finite differences
    void ICM::thermalDrift(){
      if(!sumThermalDrift) return;
      if(temperature <= real(0.0)) return;
      int numberParticles = pg->getNumberParticles();
      auto d_gridVels = thrust::raw_pointer_cast(gridVels.data());
      int BLOCKSIZE = 128;
      int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      real driftPrefactor = (dt/density)*temperature/deltaRFD;
      auto pos = pd->getPos(access::location::gpu, access::mode::read);
      ICM_ns::addThermalDrift<<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
							    (real3*)d_gridVels,
							    grid,
							    driftPrefactor,
							    deltaRFD,
							    numberParticles,
							    Kernel(grid.cellSize.x),
							    seed, step);
      CudaCheckError();
    }
    //Compute \vec{g} (except SF and thermal drift) and store it in gridVelsPrediction
    //Also stores the current advection term
    void ICM::unperturbedFluidForcing(){
      if(temperature!=real(0.0)){
	CurandSafeCall(curandSetStream(curng, st));
	sys->log<System::DEBUG2>("[Hydro::ICM] Generate random numbers");
	CurandSafeCall(curandGenerateNormal(curng, thrust::raw_pointer_cast(random.data()), random.size(), 0.0, 1.0));
      }
      real3* d_gridVels = (real3*)thrust::raw_pointer_cast(gridVels.data());
      real3* d_cellAdvection = (real3*)thrust::raw_pointer_cast(cellAdvection.data());
      auto d_random = thrust::raw_pointer_cast(random.data());
      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;
      NblocksCells.x = grid.cellDim.x/NthreadsCells.x + ((grid.cellDim.x%NthreadsCells.x)?1:0);
      NblocksCells.y = grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
      NblocksCells.z = grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
      real dV = grid.cellSize.x*grid.cellSize.y*grid.cellSize.z;
      real noiseAmp = sqrt(2*temperature*viscosity*dt/dV)/density;
      int ncells = grid.getNumberCells();
      ICM_ns::updateCellVelocityUnperturbed<<<NblocksCells, NthreadsCells, 0, st>>>(d_gridVels,
										    d_cellAdvection,
										    grid,
										    density,
										    viscosity,
										    noiseAmp,
										    dt,
										    ncells,
										    d_random);
      CudaCheckError();
    }
    //Computes \vec{\tilde{v}}=\mathcal{L}^{-1}\vec{g} in Fourier space to the current \vec{g} 
    //Stores \vec{\tilde{v}} in gridVels
    void ICM::applyStokesSolutionOperator(){
      //Stored in the same array
      auto d_gridData = (cufftReal*)thrust::raw_pointer_cast(gridVels.data());
      auto d_gridDataF = (cufftComplex*)thrust::raw_pointer_cast(gridVelsPredictionF.data());
      //Go to fourier space
      sys->log<System::DEBUG3>("[Hydro::ICM] Taking grid to wave space");
      CufftSafeCall(cufftSetStream(cufft_plan_forward, st));
      CufftSafeCall(cufftSetStream(cufft_plan_inverse, st));
      CufftSafeCall(cufftExecReal2Complex<real>(cufft_plan_forward, d_gridData, d_gridDataF));
      dim3 NthreadsCells = dim3(8,8,8);
      dim3 NblocksCells;
      NblocksCells.x = grid.cellDim.x/NthreadsCells.x + ((grid.cellDim.x%NthreadsCells.x)?1:0);
      NblocksCells.y = grid.cellDim.y/NthreadsCells.y + ((grid.cellDim.y%NthreadsCells.y)?1:0);
      NblocksCells.z = grid.cellDim.z/NthreadsCells.z + ((grid.cellDim.z%NthreadsCells.z)?1:0);
      //Solve Stokes
      sys->log<System::DEBUG3>("[Hydro::ICM] Applying Fourier Stokes operator.");
      ICM_ns::solveStokesFourier<<<NblocksCells, NthreadsCells, 0, st>>>((cufftComplex3*)d_gridDataF,
									 (cufftComplex3*)d_gridDataF,
									 viscosity,
									 density,
									 dt,
									 grid,
									 removeTotalMomemtum);
      sys->log<System::DEBUG3>("[Hydro::ICM] Going back to real space");
      //Go back to real space
      CufftSafeCall(cufftExecComplex2Real<real>(cufft_plan_inverse, d_gridDataF, d_gridData));
      CudaCheckError();
    }

    //Computes q^{n+1/2} = q^n + dt/2 J^n\vec{v}^n
    void ICM::predictorStep(){
      int numberParticles = pg->getNumberParticles();
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      {
	auto force = pd->getForce(access::location::gpu, access::mode::write);
	auto force_gr = pg->getPropertyIterator(force);
	thrust::fill(thrust::cuda::par, force_gr, force_gr + numberParticles, real4());
      }
      real3* d_gridVels = (real3*) thrust::raw_pointer_cast(gridVels.data());
      real4* d_posOld = thrust::raw_pointer_cast(posOld.data());
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      ICM_ns::midPointStep<ICM_ns::Step::PREDICTOR><<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
										  d_posOld,
										  d_gridVels,
										  dt,
										  grid,
										  Kernel(grid.cellSize.x),
										  numberParticles);
      CudaCheckError();
    }

    //Computes q^{n+1} = q^n + dt J^{n+1/2}(0.5 (\vec{v}^n+\vec{v}^n+1)
    void ICM::correctorStep(){
      int numberParticles = pg->getNumberParticles();
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      {
	auto force = pd->getForce(access::location::gpu, access::mode::write);
	auto force_gr = pg->getPropertyIterator(force);
	thrust::fill(thrust::cuda::par, force_gr, force_gr + numberParticles, real4());
      }
      real3* d_gridVels = (real3*) thrust::raw_pointer_cast(gridVels.data());
      real4* d_posOld = thrust::raw_pointer_cast(posOld.data());
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
      ICM_ns::midPointStep<ICM_ns::Step::CORRECTOR><<<Nblocks, Nthreads, 0, st>>>(pos.raw(),
										  d_posOld,
										  d_gridVels,
										  dt,
										  grid,
										  Kernel(grid.cellSize.x),
										  numberParticles);
      CudaCheckError();
    }

    void ICM::forwardTime(){
      step++;
      if(step==1){
	for(auto updatable: updatables){
	  updatable->updateTemperature(temperature);
	  updatable->updateTimeStep(dt);
	  updatable->updateViscosity(viscosity);
	  updatable->updateBox(box);
	  updatable->updateSimulationTime(0);
	}
	for(auto forceComp: interactors) forceComp->sumForce(0);
	cudaDeviceSynchronize();
      }
      sys->log<System::DEBUG1>("[Hydro::ICM] Performing integration step %d", step);
      //Take particles to t_{n+1/2}
      predictorStep();
      for(auto updatable: updatables) updatable->updateSimulationTime((step-0.5)*dt);
      //Compute unperturbed fluid forcing \vec{g}
      unperturbedFluidForcing();
      spreadParticleForces(); //Sum SF
      thermalDrift();         //Sum kT·dS/dq(q)
      //Compute \vec{\tilde{v}}^{n+1} from \vec{g}
      applyStokesSolutionOperator();
      //Given that m_e = 0, \vec{v}^{n+1} = \vec{\tilde{v}}^{n+1}
      //gridVelsPrediction = gridVels;
      //Take particles to t_{n+1}
      correctorStep();
      for(auto updatable: updatables) updatable->updateSimulationTime(step*dt);
      CudaCheckError();
    }
  }
}

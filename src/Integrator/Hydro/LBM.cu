/* Raul P. Pelaez 2018. Lattice Boltzmann Integrator.
   This file implements the D3Q19 LBM scheme with PBC using the pull-in scheme.



References:

[1] Optimized implementation of the Lattice Boltzmann Method on a graphics processing unit towards real-time fluid simulation.  N. Delbosc et al. http://dx.doi.org/10.1016/j.camwa.2013.10.002
[2] Accelerating fluid–solid simulations (Lattice-Boltzmann & Immersed-Boundary) on heterogeneous architectures. Pedro Valero-Lara et. al. https://hal.archives-ouvertes.fr/hal-01225734

 */

#include"LBM.cuh"
#include"utils/debugTools.cuh"
#include"misc/RPNG.cpp"

namespace uammd{
  namespace Hydro{
    namespace LBM{

      namespace D3Q19_ns{

        __constant__ int3 velocities[19] = {
	  { 0,-1,-1},                           //0
	  {-1, 0,-1}, {0, 0,-1}, {1, 0,-1},     //1 2 3
	  { 0, 1,-1},                           //4
	  {-1, 0, 0}, {0, 0, 0}, {1, 0, 0},     //5 6 7
	  {-1, 1, 0}, {0, 1, 0}, {1, 1, 0},     //8 9 10
	  {-1,-1, 0}, {0,-1, 0}, {1,-1, 0},     //11 12 13
	  { 0, 1, 1},                           //14
	  {-1, 0, 1}, {0, 0, 1}, {1, 0, 1},     //15 16 17
	  {0, -1, 1}};                          //18


	__constant__ int opposite[19] = {
	  14,
	  17, 16, 15,
	  18,
	  7, 6, 5,
	  13, 12, 11,
	  10, 9, 8,
	  0,
	  3, 2, 1,
	  4};


	__constant__ real wi[19] = {
	  1.0/36.0,
	  1.0/36.0, 1.0/18.0, 1.0/36.0,
	  1.0/36.0,
	  1.0/18.0, 1.0/3.0, 1.0/18.0,
	  1.0/36.0, 1.0/18.0, 1.0/36.0,
	  1.0/36.0, 1.0/18.0, 1.0/36.0,
	  1.0/36.0,
	  1.0/36.0, 1.0/18.0, 1.0/36.0,
	  1.0/36.0};


	inline __device__ real equilibriumDistribution(int i, real3 velocity, real density, real c){
	    const real eu = dot(make_real3(velocities[i]), velocity);
	    const real c2 = c*c;
	    const real si = (real(3.0)*eu/c +
	   		     real(4.5)*eu*eu/c2 -
	   		     real(1.5)/c2*dot(velocity, velocity));
	    real feq = density*wi[i]*(real(1.0) + si);

	    return feq;
	}

	__global__ void lbm_kernel(real * sourceGrid, real *destGrid,
				   int *cellType,
				   real soundSpeed,
				   real relaxTime,
				   Grid grid,
				   int ncells){
	  int id = blockIdx.x*blockDim.x + threadIdx.x;
	  if(id>=ncells) return;
	  int3 celli = make_int3(id%grid.cellDim.x,
				 (id/grid.cellDim.x)%grid.cellDim.y,
				 id/(grid.cellDim.x*grid.cellDim.y));



	  constexpr int numberVelocities = 19;
	  real density = real(0.0);
	  real3 velocity = make_real3(real(0.0));

	  real fi[numberVelocities];



	  for(int i = 0; i<numberVelocities; i++){
	    int3 cellj = celli - velocities[i];
	    cellj = grid.pbc_cell(cellj);
	    int icellj = grid.getCellIndex(cellj);
	    fi[i] = sourceGrid[icellj+i*ncells];

	    density += fi[i];
	    velocity += fi[i]*make_real3(velocities[i]);
	  }
	  velocity *= soundSpeed/density;

	  int icell = grid.getCellIndex(celli);
	  //Full way bounce-back
	  if(cellType[icell] == 2){
	    for(int i = 0; i<numberVelocities; i++){
	      destGrid[icell + i*ncells] = fi[opposite[i]];
	    }
	    return;
	  }

	  for(int i = 0; i<numberVelocities; i++){
	    real feq = equilibriumDistribution(i, velocity, density, soundSpeed);
	    destGrid[icell + i*ncells] = fi[i] -  (real(1.0)/relaxTime)*(fi[i] - feq);
	  }
	}



      __global__ void particles2Grid(int *cellType,
				     real4  *pos,
				     real  *radius,
				     int N,
				     Grid grid){
	                             //int3 P, real radius2){
	const int id = blockIdx.x;
	const int tid = threadIdx.x;
	if(id>=N) return;

	__shared__ real3 pi;
	__shared__ real radius_i;
	__shared__ int3 celli;
	if(tid==0){
	  pi = make_real3(pos[id]);
	  /*Get my cell*/
	  celli = grid.getCell(pi);
	  radius_i= real(1.0);
	  if(radius) radius_i = radius[id];
	}
	/*Conversion between cell number and cell center position*/
	const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);

	int3 P = make_int3(radius_i/grid.cellSize+0.5)+1;
	const int3 supportCells = 2*P + 1;
	const int numberNeighbourCells = supportCells.x*supportCells.y*supportCells.z;

	__syncthreads();
	for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	  /*Compute neighbouring cell*/
	  int3 cellj = make_int3(celli.x + i%supportCells.x - P.x,
				 celli.y + (i/supportCells.x)%supportCells.y - P.y,
				 celli.z + i/(supportCells.x*supportCells.y) - P.z );
	  cellj = grid.pbc_cell(cellj);
	  const int jcell = grid.getCellIndex(cellj);
	  real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize - cellPosOffset);
	  real r2 = dot(rij, rij);
	  if(r2<=(radius_i*radius_i))
	    cellType[jcell] = 2;
	}
      }









	__global__ void lbm_initial(real * sourceGrid, real *destGrid,
				    real soundSpeed,
				    real relaxTime,
				    Grid grid,
				    int ncells){
	  int id = blockIdx.x*blockDim.x + threadIdx.x;
	  if(id>=ncells) return;
	  int3 celli = make_int3(id%grid.cellDim.x,
				 (id/grid.cellDim.x)%grid.cellDim.y,
				 id/(grid.cellDim.x*grid.cellDim.y));
	  constexpr int numberVelocities = 19;
	  real density = real(1.0)/(grid.cellSize.x*grid.cellSize.y*grid.cellSize.z);
	  real3 velocity = make_real3(real(0.0));
	  int icell = grid.getCellIndex(celli);

	  for(int i = 0; i<numberVelocities; i++){
	    real feq = equilibriumDistribution(i, velocity, density, soundSpeed);
	    sourceGrid[icell + i*ncells] = feq;
	    //if(celli.x==grid.cellDim.x/2 and celli.y==grid.cellDim.y/2 and celli.z ==grid.cellDim.z/2 && i==6)    sourceGrid[icell + i*ncells] *= 10;
	    //if(celli.x==0 && i==6)    sourceGrid[icell + i*ncells] *= 10;
	    if(celli.x>grid.cellDim.x/2){
	      sourceGrid[icell + i*ncells] *= 0.5;
	    }
	    else{
	      if(i==7)
		sourceGrid[icell + i*ncells] *= 3;
	    }
	  }
	}
      }


      D3Q19::D3Q19(shared_ptr<ParticleData> pd,
		   shared_ptr<System> sys,
		   Parameters par):
	Integrator(pd, sys, "LBM::D3Q19"),
	soundSpeed(par.soundSpeed),
	relaxTime(par.relaxTime),
        viscosity(par.viscosity),
        dt(par.dt), out("fluid.dat"){
	sys->log<System::MESSAGE>("[LBM::D3Q19] Created");


	//real cellSize = soundSpeed*dt;
	//int3 cellDim = make_int3(par.box.boxSize/cellSize);
	grid = Grid(par.box, par.ncells);
	int ncells = grid.getNumberCells();



	// real cellSize = grid.cellSize.x;
	// soundSpeed = cellSize/dt;
	// relaxTime = 0.5 + 3*viscosity*dt/(cellSize*cellSize);

	sys->log<System::MESSAGE>("[LBM::D3Q19] Cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
	sys->log<System::MESSAGE>("[LBM::D3Q19] ncells: %d", ncells);
	sys->log<System::MESSAGE>("[LBM::D3Q19] soundSpeed: %e", this->soundSpeed);
	sys->log<System::MESSAGE>("[LBM::D3Q19] relaxTime: %e", this->relaxTime);
	sys->log<System::MESSAGE>("[LBM::D3Q19] viscosity: %e", viscosity);

	sourceGrid.resize(ncells*numberVelocities, 0.0);

        cellType.resize(ncells, 0);

	destGrid = sourceGrid;
	int Nthreads = 64;
	int Nblocks = ncells/Nthreads+1;
	real *sourceGrid_ptr = thrust::raw_pointer_cast(sourceGrid.data());
	real *destGrid_ptr = thrust::raw_pointer_cast(destGrid.data());

	D3Q19_ns::lbm_initial<<<Nblocks, Nthreads>>>(sourceGrid_ptr,
						     destGrid_ptr,
						     soundSpeed,
						     relaxTime,
						     grid,
						     ncells);
      }

      void D3Q19::forwardTime(){
	static int steps = 0;
	steps++;
	sys->log<System::DEBUG>("[Hydro::LBM::D3Q19] Performing step %d", steps);
	int ncells = grid.getNumberCells();
	sys->log<System::DEBUG>("[Hydro::LBM::D3Q19] Cells %d", ncells);
	int Nthreads = 64;
	int Nblocks = ncells/Nthreads+1;
	real *sourceGrid_ptr = thrust::raw_pointer_cast(sourceGrid.data());
	real *destGrid_ptr = thrust::raw_pointer_cast(destGrid.data());






	 int numberParticles = pd->getNumParticles();
	 int *cellType_ptr = thrust::raw_pointer_cast(cellType.data());
	 fillWithGPU<<<Nblocks, Nthreads>>>(cellType_ptr, 0, ncells);
	 auto pos = pd->getPos(access::location::gpu, access::mode::read);
	 auto radius = pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);



	 D3Q19_ns::particles2Grid<<<numberParticles, 32>>>(cellType_ptr,
							   pos.raw(),
							   radius.raw(),
							   numberParticles,
							   grid);

	D3Q19_ns::lbm_kernel<<<Nblocks, Nthreads>>>(sourceGrid_ptr,
						    destGrid_ptr,
						    cellType_ptr,
						    soundSpeed,
						    relaxTime/dt,
						    grid,
						    ncells);

	destGrid.swap(sourceGrid);

      }
      void D3Q19::write(){}

      void D3Q19::writePNG(){

	thrust::host_vector<real> h_data = sourceGrid;

	int ncells = grid.getNumberCells();
	std::vector<unsigned char> image(4*grid.cellDim.x*grid.cellDim.y,0);
	real max = 0;
	real min = 100000;
	fori(0, grid.cellDim.x){
	  forj(0,grid.cellDim.y){
	    //for(int kz = 0; kz<grid.cellDim.z; kz++){
	    {
	    int kz = grid.cellDim.z/2;
	    //if(kz!=20) break;
	      real density = 0.0;
	      int icell = grid.getCellIndex(make_int3(i, j, kz));
	      for(int k = 0; k<19; k++){
		density += h_data[icell+ncells*k];
	      }
	      max = std::max(max, density);
	      min = std::min(min, density);
	    }
	  }
	}

	fori(0, grid.cellDim.x){
	  forj(0,grid.cellDim.y){
	    //for(int kz = 0; kz<grid.cellDim.z; kz++){
	    {
	    int kz = grid.cellDim.z/2;
	      real density = 0.0;
	      //real3 vel = make_real3(0);
	      int icell = grid.getCellIndex(make_int3(i, j, kz));
	      for(int k = 0; k<19; k++){
		density += h_data[icell+ncells*k];

	      }
	      unsigned char R = std::min((unsigned char)255, (unsigned char)(((density-min)/(max-min))*255) );
	      unsigned char B = 255-R;
	      image[4*(i+grid.cellDim.x*j)] = R;
	      image[4*(i+grid.cellDim.x*j)+1] = 0;
	      image[4*(i+grid.cellDim.x*j)+2] = B;
	      image[4*(i+grid.cellDim.x*j)+3] = 255;
	    }
	  }
	}
	//out<<std::flush;
	static int counter = 0;
	savePNG((std::to_string(counter)+".png").c_str(), image.data(), grid.cellDim.x, grid.cellDim.y);
	counter++;
      }
    }
  }
}

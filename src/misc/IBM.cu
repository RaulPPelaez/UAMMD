#include"IBM.cuh"
#include<third_party/type_names.h>
namespace uammd{

  namespace IBM_ns{

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

      __shared__ real3 pi;
      __shared__ real3 vi; //The quantity for particle id
      __shared__ int3 celli;
      __shared__ int3 P; //Neighbour cell offset
      if(tid==0){
	pi = make_real3(pos[id]);
	vi = make_real3(v[id]);
	celli = grid.getCell(pi);

	P = make_int3(kernel.support/2);
	//Kernels with even support might need an offset of one cell depending on the position of the particle inside the cell
	if(kernel.support%2==0){
	  const real3 pi_pbc = grid.box.apply_pbc(pi);
	  P -= make_int3( (pi_pbc+grid.box.boxSize*real(0.5))*grid.invCellSize - make_real3(celli) + real(0.5) );
	}
      }
      /*Conversion between cell number and cell center position*/
      const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);
      const int supportCells = kernel.support;
      const int numberNeighbourCells = supportCells*supportCells*supportCells;
      __syncthreads();
      for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	/*Compute neighbouring cell*/
	int3 cellj = make_int3(celli.x + i%supportCells - P.x,
			       celli.y + (i/supportCells)%supportCells - P.y,
			       celli.z + i/(supportCells*supportCells) - P.z);
	cellj = grid.pbc_cell(cellj);
	  
	/*Distance from particle i to center of cell j*/
	const real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize-cellPosOffset); 
	/*The weight of particle i on cell j*/
	const real3 weight = vi*kernel.delta(rij);

	/*Get index of cell j*/
	const int jcell = grid.getCellIndex(cellj);
	  
	/*Atomically sum my contribution to cell j*/
	if(weight.x) atomicAdd(&gridVels[jcell].x, weight.x);
	if(weight.y) atomicAdd(&gridVels[jcell].y, weight.y);
	if(weight.z) atomicAdd(&gridVels[jcell].z, weight.z);
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
      //A thread per particle
      const int id = blockIdx.x*blockDim.x + threadIdx.x;
      if(id>=N) return;    
      const real3 pi = make_real3(pos[id]);
      const int3 celli = grid.getCell(pi);

      //J = S^T = St = σ S
      const real dV = (grid.cellSize.x*grid.cellSize.y*grid.cellSize.z);

      real3  result = make_real3(0);	
      int3 P = make_int3(kernel.support/2);
      //Kernels with even support might need an offset of one cell depending on the position of the particle inside the cell
      if(kernel.support%2==0){
	const real3 pi_pbc = grid.box.apply_pbc(pi);
	P -= make_int3( (pi_pbc+grid.box.boxSize*real(0.5))/grid.cellSize - make_real3(celli) + real(0.5) );
      }
      /*Transforms cell number to cell center position*/
      const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);
      const int supportCells = kernel.support;
      const int numberNeighbourCells = supportCells*supportCells*supportCells;
      for(int i = 0; i<numberNeighbourCells; i++){
	//current neighbour cell
	int3 cellj = make_int3(celli.x + i%supportCells - P.x,
			       celli.y + (i/supportCells)%supportCells - P.y,
			       celli.z + i/(supportCells*supportCells) - P.z);
	cellj = grid.pbc_cell(cellj);

	//Compute distance to cell center
	const real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize - cellPosOffset);
	const real weight = kernel.delta(rij);
	if(weight){
	  const int jcell = grid.getCellIndex(cellj);	  
	  const real3 cellj_vel = make_real3(gridVels[jcell]);
	  result += (dV*weight)*cellj_vel;
	}
      }
      //Write total to global memory
      Mv[id] += result;
    }    

    template<int TPP, class Kernel, typename vtype>
	__global__ void grid2ParticlesDTPP(const real4 * __restrict__ pos,
					   vtype * __restrict__ Mv, /*Result (i.e Mw·F)*/
					   const real3 * __restrict__ gridVels, /*Values in the grid*/
					   int N, /*Number of particles*/
					   Grid grid, /*Grid information and methods*/				  
					   Kernel kernel){
	  const int id = blockIdx.x;
	  const int tid = threadIdx.x;
	  real3  result = make_real3(0);
	  using BlockReduce = cub::BlockReduce<real3, TPP>;
	  
	  __shared__ real3 pi;
	  __shared__ int3 celli;
	  __shared__ int3 P; //Neighbour cell offset
	  __shared__ typename BlockReduce::TempStorage temp_storage;

	  if(id<N){
	  if(tid==0){
	    pi = make_real3(pos[id]);
	    celli = grid.getCell(pi);
	    
	    P = make_int3(kernel.support/2);
	    //Kernels with even support might need an offset of one cell depending on the position of the particle inside the cell
	    if(kernel.support%2==0){
	      const real3 pi_pbc = grid.box.apply_pbc(pi);
	      P -= make_int3( (pi_pbc+grid.box.boxSize*real(0.5))*grid.invCellSize - make_real3(celli) + real(0.5) );
	    }	    
	  }
	  }
	  __syncthreads();
	  if(id<N){
	  //J = S^T = St = σ S
	  const real dV = (grid.cellSize.x*grid.cellSize.y*grid.cellSize.z);

	  
	  /*Transforms cell number to cell center position*/
	  const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);
	  const int supportCells = kernel.support;
	  const int numberNeighbourCells = supportCells*supportCells*supportCells;

	  
	  for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	    //current neighbour cell
	    int3 cellj = make_int3(celli.x + i%supportCells - P.x,
				   celli.y + (i/supportCells)%supportCells - P.y,
				   celli.z + i/(supportCells*supportCells) - P.z);
	    cellj = grid.pbc_cell(cellj);

	    //Compute distance to cell center
	    const real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize - cellPosOffset);
	    const real weight = kernel.delta(rij);
	    if(weight){
	      const int jcell = grid.getCellIndex(cellj);	  
	      const real3 cellj_vel = make_real3(gridVels[jcell]);
	      result += (dV*weight)*cellj_vel;
	    }
	  }
	  }
	  
	  //Write total to global memory
	  real3 total = BlockReduce(temp_storage).Sum(result);
	  __syncthreads();
	  if(tid==0 and id<N)
	    Mv[id] += total;
	}    

  }
  
  template<class Kernel>
  IBM<Kernel>::IBM(shared_ptr<System> sys, shared_ptr<Kernel> kern):
    sys(sys), kernel(kern){
    sys->log<System::MESSAGE>("[IBM] Initialized with kernel: %s", type_name<Kernel>().c_str());
  }

  template<class Kernel>
  template<class PosIterator, class QuantityIterator>
  void IBM<Kernel>::spread(const PosIterator &pos, const QuantityIterator &v,
		      real3 *gridVels,
		      Grid grid, int numberParticles, cudaStream_t st){
    sys->log<System::DEBUG2>("[IBM] Spreading");
    //Launch a small block per particle
    {
      int support = kernel->support;
      int numberNeighbourCells = support*support*support;
      int threadsPerParticle = std::min(32*(numberNeighbourCells/32), 512);
      if(numberNeighbourCells < 64) threadsPerParticle = 32;

      IBM_ns::particles2GridD<<<numberParticles, threadsPerParticle, 0, st>>>
	(pos, v, gridVels, numberParticles, grid, *kernel);
    }


  }
  
  template<class Kernel>
  template<class PosIterator, class QuantityIterator>
  void IBM<Kernel>::gather(const PosIterator &pos, const QuantityIterator &Mv,
		      real3 *gridVels,
		      Grid grid, int numberParticles, cudaStream_t st){
    sys->log<System::DEBUG2>("[IBM] Gathering");
    
    // int BLOCKSIZE = 128;
    // int Nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
    // int Nblocks  =  numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      int support = kernel->support;
      int numberNeighbourCells = support*support*support;
      //int threadsPerParticle = std::min(32*(numberNeighbourCells/32), 512);
      int threadsPerParticle = std::min(int(pow(2,int(log2(numberNeighbourCells)+0.5))), 512);
      if(numberNeighbourCells < 64) threadsPerParticle = 32;

      auto grid2Particles = IBM_ns::grid2ParticlesDTPP<32, Kernel, real3>;

#define KERNEL(x) else if(threadsPerParticle<=x) grid2Particles = IBM_ns::grid2ParticlesDTPP<x, Kernel, real3>;
      if(threadsPerParticle<=32){}
      KERNEL(64)
	KERNEL(128)
	KERNEL(256)
	KERNEL(512)
#undef KERNEL
	
      grid2Particles<<<numberParticles, threadsPerParticle, 0, st>>>(pos, Mv, gridVels, numberParticles, grid, *kernel);



  }
    
}



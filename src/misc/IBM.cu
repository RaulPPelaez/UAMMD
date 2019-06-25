#include"IBM.cuh"
#include<third_party/type_names.h>

namespace uammd{

  namespace IBM_ns{

#ifndef SINGLE_PRECISION
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600 
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
#endif


    inline __device__ real3 atomicAdd(real3* address, real3 val){
      real3 newval;
      if(val.x) newval.x = atomicAdd(&(*address).x, val.x);
      if(val.y) newval.y = atomicAdd(&(*address).y, val.y);
      if(val.z) newval.z = atomicAdd(&(*address).z, val.z);
      return newval;
    }

    inline __device__ real2 atomicAdd(real2* address, real2 val){
      real2 newval;
      if(val.x) newval.x = atomicAdd(&(*address).x, val.x);
      if(val.y) newval.y = atomicAdd(&(*address).y, val.y);
      return newval;
    }

    /*Spreads the 3D quantity v (defined on the particle positions) to a grid
      
      S v(z) = v(x) = \sum_{z}{ v(z)*\delta(||z-x||^2) }
      Where:
      - S is the spreading operator
      - "v" is a certain quantity
      - "z" is the position of the particles
      - "x" is the position of a grid cell
      - \delta() is the window function
    */
    template<bool is2D, class Grid, class Kernel,
      class PosIterator,
      class ParticleQuantityIterator, class GridQuantityIterator>
    __global__ void particles2GridD(const PosIterator __restrict__ pos, /*Particle positions*/
				    const ParticleQuantityIterator __restrict__ v,   /*Per particle quantity to spread*/
				    GridQuantityIterator  __restrict__ gridQuantity, /*Spreaded values, size ncells*/
				    int N, /*Number of particles*/
				    Grid grid, /*Grid information and methods*/
				    Kernel kernel){
      const int id = blockIdx.x;
      const int tid = threadIdx.x;
      using QuantityType = typename std::iterator_traits<GridQuantityIterator>::value_type;
      using ParticleQuantityType = typename std::iterator_traits<ParticleQuantityIterator>::value_type;
      
      if(id>=N) return;

      __shared__ real3 pi;
      __shared__ ParticleQuantityType vi; //The quantity for particle id
      __shared__ int3 celli;
      __shared__ int3 P; //Neighbour cell offset
      if(tid==0){
	pi = make_real3(pos[id]);
	vi = v[id];
	celli = grid.getCell(pi);
	const auto invCellSize = real(1.0)/grid.getCellSize(celli);
	  
	P = make_int3(kernel.support/2);
	//Kernels with even support might need an offset of one cell depending on the position of the particle inside the cell
	if(kernel.support%2==0){
	  const real3 pi_pbc = grid.box.apply_pbc(pi);
	  P -= make_int3( (pi_pbc+grid.box.boxSize*real(0.5))*invCellSize - make_real3(celli) + real(0.5) );
	}
	if(is2D) P.z = 0;
      }
      const int supportCells = kernel.support;
      int numberNeighbourCells = supportCells*supportCells;
      if(!is2D)  numberNeighbourCells *= supportCells;
      __syncthreads();
      for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	/*Compute neighbouring cell*/
	int3 cellj = make_int3(celli.x + i%supportCells - P.x,
			       celli.y + (i/supportCells)%supportCells - P.y,
			       is2D?0:(celli.z + i/(supportCells*supportCells) - P.z));
	cellj = grid.pbc_cell(cellj);
	  
	/*Distance from particle i to center of cell j*/
	const real3 rij = grid.distanceToCellCenter(pi, cellj);
	const real k = kernel.delta(rij, grid.getCellSize(cellj));
	/*The weight of particle i on cell j*/
	const auto weight = vi*kernel.delta(rij, grid.getCellSize(cellj));
	// if(weight.x) printf("celli: %d %d , cellj: %d %d , weight: %g %g\n",
	// 		    celli.x, celli.y,
	// 		    cellj.x, cellj.y,
	// 		    weight.x, weight.y);
	/*Get index of cell j*/
	const int jcell = grid.getCellIndex(cellj);
	  
	/*Atomically sum my contribution to cell j*/
	atomicAdd(&gridQuantity[jcell], weight);
      }
    }

    /*Interpolates a quantity (i.e velocity) from its values in the grid to the particles.

      J(z) q(x) = q(z) = \sum_{x\in G}{ q(x)*\delta(||x-z||^2) weight(x)}
      Where :
         - J is the interpolation operator
	 - "q" a certain quantity
	 - "x" a cell of the grid
	 - "z" the position of a particle 
	 - \delta() is the window function given by Kernel
	 - weight() is the quadrature weight of a cell. (cellsize^d in a regular grid) given by QuadratureWeights

      This is the discretization of an integral and thus requires quadrature weigths for each element. 
        Which in a regular grid is just the cell size, h. But can in general be something depending on the position.
    */

    template<int TPP, bool is2D, class Grid,
      class Kernel,
      class PosIterator, class ResultIterator, class GridQuantityIterator,
      class QuadratureWeights>
    __global__ void grid2ParticlesDTPP(const PosIterator pos, /*Particle positions*/
				       ResultIterator Jq, /*Result for each particle*/
				       const GridQuantityIterator gridQuantity, /*Values in the grid*/
				       int N, /*Number of particles*/
				       Grid grid, /*Grid information and methods*/				  
				       Kernel kernel, 
				       QuadratureWeights qw /*Quadrature weights*/
				       ){
      const int id = blockIdx.x;
      const int tid = threadIdx.x;
      
      
      using GridQuantityType = typename std::iterator_traits<GridQuantityIterator>::value_type;
      using BlockReduce = cub::BlockReduce<GridQuantityType, TPP>;

      GridQuantityType result = GridQuantityType();
      
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
	    const real3 invCellSize = real(1.0)/grid.getCellSize(celli);
	    const real3 pi_pbc = grid.box.apply_pbc(pi);
	    P -= make_int3( (pi_pbc+grid.box.boxSize*real(0.5))*invCellSize - make_real3(celli) + real(0.5) );
	  }
	  if(is2D) P.z = 0;
	}
      }
      __syncthreads();
      if(id<N){
	const int supportCells = kernel.support;
	int numberNeighbourCells = supportCells*supportCells;
	if(!is2D)  numberNeighbourCells *= supportCells;
	
	for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	  //current neighbour cell
	  int3 cellj = make_int3(celli.x + i%supportCells - P.x,
				 celli.y + (i/supportCells)%supportCells - P.y,
				 is2D?0:(celli.z + i/(supportCells*supportCells) - P.z));
	  cellj = grid.pbc_cell(cellj);

	  //Compute distance to cell center
	  const real3 rij = grid.distanceToCellCenter(pi, cellj);
	  
	  const real weight = kernel.delta(rij, grid.getCellSize(cellj));

	  if(weight){
	    //J = S^T = St = σ S 
	    const int jcell = grid.getCellIndex(cellj);
	    const auto cellj_vel = gridQuantity[jcell];
	    const real dV = qw(cellj, grid);
	    // printf("i: %d ; cellj: %d %d %d ; celli: %d %d %d; rij: %g %g %g; weight: %.17g ; dV: %g\n",
	    //  	   id, cellj.x, cellj.y, cellj.z, celli.x, celli.y, celli.z, rij.x, rij.y, rij.z, weight, dV);
	    result += (dV*weight)*cellj_vel;
	  }
	}
      }
	  
      //Write total to global memory
      GridQuantityType total = BlockReduce(temp_storage).Sum(result);
      __syncthreads();
      if(tid==0 and id<N){
	using ResultType = typename std::iterator_traits<ResultIterator>::value_type;
	Jq[id] += static_cast<ResultType>(total);
      }
    }

  }
  
  template<class Kernel>
  IBM<Kernel>::IBM(shared_ptr<System> sys, shared_ptr<Kernel> kern):
    sys(sys), kernel(kern){
    sys->log<System::MESSAGE>("[IBM] Initialized with kernel: %s", type_name<Kernel>().c_str());
  }

  template<class Kernel>
  template<class Grid, class PosIterator, class QuantityIterator, class GridDataIterator>
  void IBM<Kernel>::spread(const PosIterator &pos, const QuantityIterator &v,
			   GridDataIterator &gridVels,
			   Grid grid, int numberParticles, cudaStream_t st){
    sys->log<System::DEBUG2>("[IBM] Spreading");
    //Launch a small block per particle
    {
      int support = kernel->support;
      int numberNeighbourCells = support*support*support;
      int threadsPerParticle = std::min(32*(numberNeighbourCells/32), 512);
      if(numberNeighbourCells < 64) threadsPerParticle = 32;

      if(grid.cellDim.z == 1)
	IBM_ns::particles2GridD<true><<<numberParticles, threadsPerParticle, 0, st>>>
	  (pos, v, gridVels, numberParticles, grid, *kernel);
      else
	IBM_ns::particles2GridD<false><<<numberParticles, threadsPerParticle, 0, st>>>
	  (pos, v, gridVels, numberParticles, grid, *kernel);

    }
  }

  namespace IBM_ns{
    struct DefaultQuadratureWeights{
      inline __host__ __device__ real operator()(int3 cellj, const Grid &grid) const{
	return grid.getCellVolume(cellj);
      }
    };
  }
  template<class Kernel>
  template<class Grid,
      class PosIterator, class ResultIterator, class GridQuantityIterator>
  void IBM<Kernel>::gather(const PosIterator &pos, const ResultIterator &Jq,
			   const GridQuantityIterator &gridData,
			   Grid & grid, int numberParticles, cudaStream_t st){
    IBM_ns::DefaultQuadratureWeights qw;
    this->gather(pos, Jq, gridData, grid, qw, numberParticles, st);
  }
  template<class Kernel>
  template<class Grid,
    class PosIterator, class ResultIterator, class GridQuantityIterator,
    class QuadratureWeights>
  void IBM<Kernel>::gather(const PosIterator &pos, const ResultIterator &Jq,
			   const GridQuantityIterator &gridData,
			   Grid & grid, const QuadratureWeights &qw, int numberParticles, cudaStream_t st){
    if(grid.cellDim.z == 1)
      gather<true>(pos, Jq, gridData, grid, qw, numberParticles, st);    
    else
      gather<false>(pos, Jq, gridData, grid, qw, numberParticles, st);
    

  }

  template<class Kernel>
  template<bool is2D, class Grid,
    class PosIterator, class ResultIterator, class GridQuantityIterator,
    class QuadratureWeights>
  void IBM<Kernel>::gather(const PosIterator &pos, const ResultIterator &Jq,
			   const GridQuantityIterator &gridData,
			   Grid & grid, const QuadratureWeights &qw, int numberParticles, cudaStream_t st){
    sys->log<System::DEBUG2>("[IBM] Gathering");
    
    int support = kernel->support;
    int numberNeighbourCells = support*support*support;
    int threadsPerParticle = std::min(int(pow(2,int(std::log2(numberNeighbourCells)+0.5))), 512);
    if(numberNeighbourCells < 64) threadsPerParticle = 32;

    auto grid2Particles = IBM_ns::grid2ParticlesDTPP<32, is2D, Grid, Kernel, PosIterator,  ResultIterator, GridQuantityIterator, QuadratureWeights>;

#define KERNEL(x) else if(threadsPerParticle<=x) grid2Particles = IBM_ns::grid2ParticlesDTPP<x, is2D, Grid, Kernel, PosIterator, ResultIterator, GridQuantityIterator, QuadratureWeights>;
    if(threadsPerParticle<=32){}
      KERNEL(64)
      KERNEL(128)
      KERNEL(256)
      KERNEL(512)
#undef KERNEL
	
      grid2Particles<<<numberParticles, threadsPerParticle, 0, st>>>(pos, Jq, gridData,
								     numberParticles, grid, *kernel, qw);



  }
  
}



#include"IBM.cuh"
#include<third_party/type_names.h>

namespace uammd{

  namespace IBM_ns{

    template<class T>
    inline __device__ T atomicAdd(T* address, T val){ return ::atomicAdd(address, val);}

#ifndef SINGLE_PRECISION
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
      inline __device__ double atomicAdd(double* address, double val){
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
    template<bool is2D, class Grid, class Index3D, class Kernel,
      class PosIterator,
      class ParticleQuantityIterator, class GridQuantityIterator>
    __global__ void particles2GridD(const PosIterator pos, /*Particle positions*/
				    const ParticleQuantityIterator  v,   /*Per particle quantity to spread*/
				    GridQuantityIterator  __restrict__ gridQuantity, /*Spreaded values, size ncells*/
				    int N, /*Number of particles*/
				    Grid grid, /*Grid information and methods*/
				    Index3D cell2index,
				    Kernel kernel){
      const int id = blockIdx.x;
      const int tid = threadIdx.x;
      using QuantityType = typename std::iterator_traits<GridQuantityIterator>::value_type;
      using ParticleQuantityType = typename std::iterator_traits<ParticleQuantityIterator>::value_type;
      if(id>=N) return;
      __shared__ real3 pi;
      __shared__ ParticleQuantityType vi; 
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
	int3 cellj = make_int3(celli.x + i%supportCells - P.x,
			       celli.y + (i/supportCells)%supportCells - P.y,
			       is2D?0:(celli.z + i/(supportCells*supportCells) - P.z));
	cellj = grid.pbc_cell(cellj);
	const real3 rij = grid.distanceToCellCenter(pi, cellj);
	const auto weight = vi*kernel.delta(rij, grid.getCellSize(cellj));
	const int jcell = cell2index(cellj);
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
      class Index3D,
      class QuadratureWeights>
    __global__ void grid2ParticlesDTPP(PosIterator pos, /*Particle positions*/
				       ResultIterator Jq, /*Result for each particle*/
				       GridQuantityIterator gridQuantity, /*Values in the grid*/
				       int N, /*Number of markers*/
				       Grid grid,
				       Index3D cell2index,
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
	  const real3 rij = grid.distanceToCellCenter(pi, cellj);
	  const auto weight = kernel.delta(rij, grid.getCellSize(cellj));
	  const int jcell = cell2index(cellj);
	  const auto cellj_vel = gridQuantity[jcell];
	  const real dV = qw(cellj, grid);
	  result += (dV*weight)*cellj_vel;
	}
      }
      GridQuantityType total = BlockReduce(temp_storage).Sum(result);
      __syncthreads();
      if(tid==0 and id<N){
	using ResultType = typename std::iterator_traits<ResultIterator>::value_type;
	Jq[id] += static_cast<ResultType>(total);
      }
    }

    template<int threadsPerParticle, bool is2D, class ...T> void callGather(int numberParticles, cudaStream_t st, T... args){
      grid2ParticlesDTPP<threadsPerParticle, is2D><<<numberParticles, threadsPerParticle, 0, st>>>(args...);
    }
    
  }
}



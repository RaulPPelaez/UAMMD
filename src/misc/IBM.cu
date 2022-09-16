/*Raul P. Pelaez 2019-2021. Immersed Boundary Method (IBM).
  See IBM.cuh
 */
#include<type_traits>
#include<third_party/uammd_cub.cuh>
#include "utils/atomics.cuh"
#include "IBM_utils.cuh"
namespace uammd{
  namespace IBM_ns{
    namespace detail{
      template<class Grid>
      __device__ int3 computeSupportShift(real3 pos, int3 celli, Grid grid, int3 support){
	int3 P = support/2;
	//Sometimes we need to displace the support cells to the left
	const auto distanceToLeftMostCell = abs(grid.distanceToCellCenter(pos, celli-P));
	const auto cellSize = grid.getCellSize(celli);
	if(cellSize.x>0 and distanceToLeftMostCell.x > support.x*cellSize.x/real(2.0)){
	  P.x -=1;
	}
	if(cellSize.y>0 and distanceToLeftMostCell.y > support.y*cellSize.y/real(2.0)){
	  P.y -=1;
	}
	if(cellSize.z>0 and distanceToLeftMostCell.z > support.z*cellSize.z/real(2.0)){
	  P.z -=1;
	}
	return P;
      }

      template<class Grid, class Kernel, class KernelValueType>
      __device__ void fillSharedWeights(KernelValueType* weights, real3 pi, int3 support, int3 celli, int3 P,  Grid &grid, Kernel &kernel){
	auto *weightsX = &weights[0];
	const int tid = threadIdx.x;
	for(int i = tid; i<support.x; i+=blockDim.x){
	  const auto cellj = make_int3(grid.pbc_cell_coord<0>(celli.x + i - P.x), celli.y, celli.z);
	  if(cellj.x>=0){
	    const real rij = grid.distanceToCellCenter(pi, cellj).x;
	    weightsX[i] = detail::phiX(kernel, rij, pi);
	  }
	}
	auto *weightsY = &weights[support.x];
	for(int i = tid; i<support.y; i+=blockDim.x){
	  const auto cellj = make_int3(celli.x, grid.pbc_cell_coord<1>(celli.y + i -P.y), celli.z);
	  if(cellj.y>=0){
	    const real rij = grid.distanceToCellCenter(pi, cellj).y;
	    weightsY[i] = detail::phiY(kernel, rij, pi);
	  }
	}
	auto *weightsZ = &weights[support.x+support.y];
	for(int i = tid; i<support.z; i+=blockDim.x){
	  const auto cellj = make_int3(celli.x, celli.y, grid.pbc_cell_coord<2>(celli.z + i - P.z));
	  if(cellj.z>=0){
	    const real rij = grid.distanceToCellCenter(pi, cellj).z;
	    weightsZ[i] = detail::phiZ(kernel, rij, pi);
	  }
	}
      }

      __device__ real3 computeWeightFromShared(real* weights, int ii, int jj, int kk, int3 support){
	return make_real3(weights[ii], weights[support.x+jj], weights[support.x+support.y+kk]);
      }
    }
    /*Spreads the quantity v (defined on the particle positions) to a grid
      S v(z) = v(x) = \sum_{z}{ v(z)*\delta(z-x) }
      Where:
      - S is the spreading operator
      - "v" is a certain quantity
      - "z" is the position of the particles
      - "x" is the position of a grid cell
      - \delta() is the window function
    */
    template<bool is2D, class Grid, class Index3D, class Kernel,
	     class PosIterator,
	     class ParticleQuantityIterator, class GridQuantityIterator, class WeightCompute>
    __global__ void particles2GridD(const PosIterator pos,
				    const ParticleQuantityIterator  particleQuantity,
				    GridQuantityIterator  __restrict__ gridQuantity,
				    int numberParticles,
				    Grid grid,
				    Index3D cell2index,
				    Kernel kernel,
				    WeightCompute weightCompute){
      const int id = blockIdx.x;
      const int tid = threadIdx.x;
      using GridQuantityType = typename std::iterator_traits<GridQuantityIterator>::value_type;
      using ParticleQuantityType = typename std::iterator_traits<ParticleQuantityIterator>::value_type;
      if(id>=numberParticles) return;
      __shared__ real3 pi;
      __shared__ ParticleQuantityType vi;
      __shared__ int3 celli;
      __shared__ int3 P; //Neighbour cell offset
      __shared__ int3 support;
      using KernelValueType = decltype(detail::phiX(kernel,real(), real3()));
      extern __shared__ KernelValueType weights[];
      if(tid==0){
	pi = make_real3(pos[id]);
	vi = particleQuantity[id];
	celli = grid.getCell(pi);
	support = detail::GetSupport<Kernel>::get(kernel, pi, celli);
	P = detail::computeSupportShift(pi, celli, grid, support);
	if(is2D){
	  P.z = 0;
	  support.z = 1;
	}
      }
      __syncthreads();
      detail::fillSharedWeights(weights, pi, support, celli, P, grid, kernel);
      __syncthreads();
      int numberNeighbourCells = support.x*support.y;
      if(!is2D)  numberNeighbourCells *= support.z;
      for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	const int ii = i%support.x;
	const int jj=(i/support.x)%support.y;
	const int kk=is2D?0:(i/(support.x*support.y));
	const int3 cellj = grid.pbc_cell(make_int3(celli.x + ii - P.x, celli.y + jj - P.y, is2D?0:(celli.z + kk - P.z)));
	if(cellj.x<0 or cellj.y <0 or cellj.z<0)
	  continue;
	const int jcell = cell2index(cellj);
	const auto kern = detail::computeWeightFromShared(weights, ii, jj, kk, support);
	const auto weight = weightCompute(vi,kern);
        atomicAdd(gridQuantity[jcell], weight);
      }
    }

    /*Interpolates a quantity (i.e velocity) from its values in the grid to the particles.
      J(z) q(x) = q(z) = \sum_{x\in G}{ q(x)*\delta(x-z) weight(x)}
      Where :
         - J is the interpolation operator
	 - "q" a certain quantity
	 - "x" a cell of the grid
	 - "z" the position of a particle
	 - \delta() is the window function given by Kernel
	 - weight() is the quadrature weight of a cell. (cellsize^d in a regular grid) given by QuadratureWeights

      This is the discretization of an integral and thus requires quadrature weigths for each element.
        Which in a regular grid is just the cell size, h. But can in general be something that depends on the position.
    */
    template<int TPP, bool is2D, class Grid,
      class InterpolationKernel,
      class ParticlePosIterator, class ParticleQuantityOutputIterator,
      class GridQuantityIterator,
      class WeightCompute,
      class Index3D,
      class QuadratureWeights>
    __global__ void grid2ParticlesDTPP(const ParticlePosIterator pos,
				       ParticleQuantityOutputIterator particleQuantity,
				       const GridQuantityIterator gridQuantity,
				       int numberParticles,
				       Grid grid,
				       Index3D cell2index, //Index of a 3d cell in gridQuantity
				       InterpolationKernel kernel,
				       WeightCompute weightCompute,
				       QuadratureWeights qw){
      const int id = blockIdx.x;
      const int tid = threadIdx.x;
      using GridQuantityType = typename std::iterator_traits<GridQuantityIterator>::value_type;
      using ParticleQuantityType = typename std::iterator_traits<ParticleQuantityOutputIterator>::value_type;
      using BlockReduce = cub::BlockReduce<GridQuantityType, TPP>;
      GridQuantityType result = GridQuantityType();
      __shared__ real3 pi;
      __shared__ int3 celli;
      __shared__ int3 P; //Neighbour cell offset
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ int3 support;
      using KernelValueType = decltype(detail::phiX(kernel,real(), real3()));
      extern __shared__ KernelValueType weights[];
      if(id<numberParticles){
	if(tid==0){
	  pi = make_real3(pos[id]);
	  celli = grid.getCell(pi);
	  support = detail::GetSupport<InterpolationKernel>::get(kernel, pi, celli);
	  P = detail::computeSupportShift(pi, celli, grid, support);
	  if(is2D){
	    P.z = 0;
	    support.z = 1;
	  }
	}
      }
      __syncthreads();
      detail::fillSharedWeights(weights, pi, support, celli, P, grid, kernel);
      __syncthreads();
      if(id<numberParticles){
	int numberNeighbourCells = support.x*support.y;
	if(!is2D)  numberNeighbourCells *= support.z;
	for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	  const int ii = i%support.x;
	  const int jj=(i/support.x)%support.y;
	  const int kk=is2D?0:(i/(support.x*support.y));
	  const int3 cellj = grid.pbc_cell(make_int3(celli.x + ii - P.x, celli.y + jj - P.y, is2D?0:(celli.z + kk - P.z)));
	  if(cellj.x<0 or cellj.y <0 or cellj.z<0)
	    continue;
	  const real dV = qw(cellj, grid);
	  const int jcell = cell2index(cellj);
	  const auto kern = detail::computeWeightFromShared(weights, ii, jj, kk, support);
	  const auto weight = weightCompute(gridQuantity[jcell], kern);
	  result += dV*weight;
	}
      }
      GridQuantityType total = BlockReduce(temp_storage).Sum(result);
      if(tid==0 and id<numberParticles){
	particleQuantity[id] += total;
      }
    }

    template<int threadsPerParticle, bool is2D, class ...T> void callGather(int numberParticles, size_t shMemory, cudaStream_t st, T... args){
      grid2ParticlesDTPP<threadsPerParticle, is2D><<<numberParticles, threadsPerParticle, shMemory, st>>>(args...);
    }

  }
}

/*Raul P. Pelaez 2019-2020. Immersed Boundary Method (IBM).
This class contains functions to spread marker information to a grid and
interpolate information from a grid to some marker positions. This can be
adapted to aid in, for example, an Immersed Boundary Method or a NUFFT method.

It allows to employ any Kernel to do so, see IBM_kernels.cuh.
Furthemore the quadrature weights can also be specified when interpolating.

A Kernel has the following requirements:

  -A publicly accesible member called support (either an int or int3) or a
function getSupport(int3 cell) if support depends on grid position
  -A function phi(real r) that returns the window function evaluated at that distance

USAGE:

//Creation, needs the System shared_ptr
using Kernel = IBM_kernels::PeskinKernel::threePoint;
auto ibm = std::make_shared<IBM<Kernel>>(sys, kernel);

//Spread to a grid

ibm->spread(pos,      //An iterator with the position of the markers
            quantity, //An iterator with the quantity of each marker to spread
	    gridQuantity, //An iterator with the grid data (ibm will sum to the existing data)
	    grid,         //A Grid descriptor corresponding to gridQuantity
	    numberMarkers,
	    cudaStream);

//Interpolate from a grid

ibm->gather(pos,      //An iterator with the position of the markers
            quantity, //An iterator with the quantity of each marker to gather (ibm will sum to the existing values)
	    gridQuantity, //An iterator with the grid data
	    grid,         //A Grid descriptor corresponding to gridQuantity
	    //qw,  //Optional, a device functor that takes (cell, grid) and returns the quadrature weight of cell. cellVolume() is the default.
	    numberMarkers,
	    cudaStream);

//The value types of the different iterators can be wildly different as long as they are compatible, in the sense that the type of quantity*delta() must be summable to the type of gridQuantity for spread, etc.


//Get a reference to the kernel
auto kernel = ibm->getKernel();

REFERENCES:
[1] Charles S. Peskin. The immersed boundary method (2002). DOI: 10.1017/S0962492902000077
*/
#ifndef MISC_IBM_CUH
#define MISC_IBM_CUH
#include"global/defines.h"
#include"utils/utils.h"
#include"System/System.h"
#include "IBM.cu"
#include "IBM_utils.cuh"
#include "utils/Grid.cuh"

namespace uammd{
  namespace IBM_ns{
    struct LinearIndex3D{
      LinearIndex3D(int nx, int ny, int nz):nx(nx), ny(ny), nz(nz){}

      inline __device__ __host__ int operator()(int3 c) const{
	return this->operator()(c.x, c.y, c.z);
      }

      inline __device__ __host__ int operator()(int i, int j, int k) const{
	return i + nx*(j+ny*k);
      }

    private:
      const int nx, ny, nz;
    };

    struct DefaultQuadratureWeights{
      template<class Grid>
      inline __host__ __device__ real operator()(int3 cellj, const Grid &grid) const{
	return grid.getCellVolume(cellj);
      }
    };

    struct DefaultWeightCompute{
      template<class T1, class T2>
      inline __device__ auto operator()(T1 value, thrust::tuple<T2,T2,T2> kernel) const{
	auto phiX = thrust::get<0>(kernel);
	auto phiY = thrust::get<1>(kernel);
	auto phiZ = thrust::get<2>(kernel);
	return value*phiX*phiY*phiZ;
      }
    };
  }
  template<class Kernel, class Grid = uammd::Grid, class Index3D = IBM_ns::LinearIndex3D>
  class IBM{
    shared_ptr<Kernel> kernel;
    Grid grid;
    Index3D cell2index;
  public:

    IBM(shared_ptr<Kernel> kern, Grid a_grid, Index3D cell2index):
      kernel(kern), grid(a_grid), cell2index(cell2index){
      System::log<System::DEBUG2>("[IBM] Initialized with kernel: %s", type_name<Kernel>().c_str());
    }

    IBM(shared_ptr<Kernel> kern, Grid a_grid):
      IBM(kern, a_grid, Index3D(a_grid.cellDim.x, a_grid.cellDim.y,a_grid.cellDim.z )){}

    template<bool is2D, class PosIterator, class QuantityIterator,
	     class GridDataIterator, class WeightCompute = IBM_ns::DefaultWeightCompute>
    void spread(const PosIterator &pos, const QuantityIterator &v,
		GridDataIterator &gridData,
		WeightCompute &weightCompute,
		int numberParticles, cudaStream_t st = 0) const{
      System::log<System::DEBUG2>("[IBM] Spreading");
      int3 support = IBM_ns::detail::GetMaxSupport<Kernel>::get(*kernel);
      int numberNeighbourCells = support.x*support.y*((is2D?1:support.z));
      int threadsPerParticle = std::min(32*(numberNeighbourCells/32), 128);
      if(numberNeighbourCells < 64){
	threadsPerParticle = 32;
      }
      using KernelValueType = decltype(IBM_ns::detail::phiX(*kernel,real(), real3()));
      size_t shMemory = (support.x+support.y+(!is2D)*support.z)*sizeof(KernelValueType);
      IBM_ns::particles2GridD<is2D><<<numberParticles, threadsPerParticle, shMemory, st>>>
	(pos, v, gridData, numberParticles, grid, cell2index, *kernel, weightCompute);
    }

    template<bool is2D,
      class PosIterator, class ResultIterator, class GridQuantityIterator,
	     class QuadratureWeights, class WeightCompute>
    void gather(const PosIterator &pos, ResultIterator &Jq,
		const GridQuantityIterator &gridData,
		const QuadratureWeights &qw,
		const WeightCompute &wc,
		int numberParticles, cudaStream_t st = 0) const{
      System::log<System::DEBUG2>("[IBM] Gathering");
      int3 support = IBM_ns::detail::GetMaxSupport<Kernel>::get(*kernel);
      int numberNeighbourCells = support.x*support.y*((is2D?1:support.z));
      int threadsPerParticle = std::min(int(pow(2,int(std::log2(numberNeighbourCells)+0.5))), 64);
      using KernelValueType = decltype(IBM_ns::detail::phiX(*kernel,real(), real3()));
      size_t shMemory = (support.x+support.y+(!is2D)*support.z)*sizeof(KernelValueType);
      if(numberNeighbourCells < 64){
	threadsPerParticle = 32;
      }
#define KERNEL(x) if(threadsPerParticle<=x){ \
	              IBM_ns::callGather<x, is2D>(numberParticles,      \
						  shMemory, st,		\
						  pos, Jq, gridData,	\
						  numberParticles,	\
						  grid, cell2index,	\
						  *kernel, wc, qw);	\
		      return;						\
                  }
      KERNEL(32)
	KERNEL(64)
#undef KERNEL

    }

    template<class ...T>
    void gather(T... args) const{
      if(grid.cellDim.z == 1)
	gather<true>(args...);
      else
	gather<false>(args...);
    }

    template<class PosIterator, class ResultIterator, class GridQuantityIterator>
    void gather(PosIterator pos, ResultIterator Jq,
	        GridQuantityIterator gridData,
		int numberParticles, cudaStream_t st = 0) const{
      IBM_ns::DefaultQuadratureWeights qw;
      IBM_ns::DefaultWeightCompute wc;
      this->gather(pos, Jq, gridData, qw, wc, numberParticles, st);
    }

    template<class ...T>
    void spread(T... args) const{
      if(grid.cellDim.z == 1)
	spread<true>(args...);
      else
	spread<false>(args...);
    }

    template<bool is2D, class PosIterator, class QuantityIterator,
	     class GridDataIterator>
    void spread(const PosIterator &pos, const QuantityIterator &v,
		GridDataIterator &gridData,
		int numberParticles, cudaStream_t st = 0) const{
      IBM_ns::DefaultWeightCompute wc;
      spread(pos, v, gridData, wc, numberParticles, st);
    }

    shared_ptr<Kernel> getKernel(){ return this->kernel;}

  };

}

#endif

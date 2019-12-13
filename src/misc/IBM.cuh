/*Raul P. Pelaez 2019. Immersed Boundary Method (IBM).
This class contains functions to spread marker information to a grid and interpolate information from a grid to some marker positions. This can be adapted to aid in, for example, an Immersed Boundary Method or a NUFFT method.

It allows to employ any Kernel to do so, see IBM_kernels.cuh.
Furthemore the quadrature weights can also be specified when interpolating.

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
#include"IBM.cu"

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
      inline __host__ __device__ real operator()(int3 cellj, const Grid &grid) const{
	return grid.getCellVolume(cellj);
      }
    };

  }
  
  template<class Kernel, class Grid = uammd::Grid, class Index3D = IBM_ns::LinearIndex3D>
  class IBM{
    shared_ptr<Kernel> kernel;
    shared_ptr<System> sys;
    Grid grid;
    Index3D cell2index;
  public:
    
    IBM(shared_ptr<System> sys, shared_ptr<Kernel> kern, Grid a_grid, Index3D cell2index):      
      sys(sys), kernel(kern), grid(a_grid), cell2index(cell2index){
      sys->log<System::MESSAGE>("[IBM] Initialized with kernel: %s", type_name<Kernel>().c_str());
    }
    
    IBM(shared_ptr<System> sys, shared_ptr<Kernel> kern, Grid a_grid):
      IBM(sys, kern, a_grid, Index3D(a_grid.cellDim.x, a_grid.cellDim.y,a_grid.cellDim.z )){}

    template<class PosIterator, class QuantityIterator, class GridDataIterator>
    void spread(const PosIterator &pos, const QuantityIterator &v,
		GridDataIterator &gridData,
		int numberParticles, cudaStream_t st = 0){
      sys->log<System::DEBUG2>("[IBM] Spreading");
      int support = kernel->support;
      int numberNeighbourCells = support*support*support;
      int threadsPerParticle = std::min(32*(numberNeighbourCells/32), 512);
      if(numberNeighbourCells < 64){
	threadsPerParticle = 32;
      }
      if(grid.cellDim.z == 1){
	IBM_ns::particles2GridD<true><<<numberParticles, threadsPerParticle, 0, st>>>
	  (pos, v, gridData, numberParticles, grid, cell2index, *kernel);
      }
      else{
	IBM_ns::particles2GridD<false><<<numberParticles, threadsPerParticle, 0, st>>>
	  (pos, v, gridData, numberParticles, grid, cell2index, *kernel);
      }
    }

    template<class PosIterator, class ResultIterator, class GridQuantityIterator>
    void gather(const PosIterator &pos, const ResultIterator &Jq,
		const GridQuantityIterator &gridData,
		int numberParticles, cudaStream_t st = 0){
      IBM_ns::DefaultQuadratureWeights qw;
      this->gather(pos, Jq, gridData, qw, numberParticles, st);
    }

    template<class PosIterator, class ResultIterator, class GridQuantityIterator,
      class QuadratureWeights>
    void gather(const PosIterator &pos, const ResultIterator &Jq,
		const GridQuantityIterator &gridData,
		const QuadratureWeights &qw, int numberParticles, cudaStream_t st = 0){
      if(grid.cellDim.z == 1)
	gather<true>(pos, Jq, gridData, qw, numberParticles, st);
      else
	gather<false>(pos, Jq, gridData, qw, numberParticles, st);
    }

    template<bool is2D,
      class PosIterator, class ResultIterator, class GridQuantityIterator,
      class QuadratureWeights>
    void gather(const PosIterator &pos, const ResultIterator &Jq,
		const GridQuantityIterator &gridData,
		const QuadratureWeights &qw, int numberParticles, cudaStream_t st = 0){
      sys->log<System::DEBUG2>("[IBM] Gathering");
      int support = kernel->support;
      int numberNeighbourCells = support*support*support;
      int threadsPerParticle = std::min(int(pow(2,int(std::log2(numberNeighbourCells)+0.5))), 512);
      if(numberNeighbourCells < 64){
	threadsPerParticle = 32;
      }
#define KERNEL(x) if(threadsPerParticle<=x){ IBM_ns::callGather<x, is2D>(numberParticles, st, pos, Jq, gridData, numberParticles, grid, cell2index, *kernel, qw); return;}
      KERNEL(32)
	KERNEL(64)
	KERNEL(128)
	KERNEL(256)
	KERNEL(512)
#undef KERNEL

    }

    shared_ptr<Kernel> getKernel(){ return this->kernel;}

  };

}

#endif
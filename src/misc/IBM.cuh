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
	    gridQuantity, //A real3* with the grid data (ibm will sum to the existing data)
	    grid,         //A Grid descriptor corresponding to gridQuantity
	    numberMarkers,
	    cudaStream);

//Interpolate from a grid

ibm->gather(pos,      //An iterator with the position of the markers
            quantity, //An iterator with the quantity of each marker to gather (ibm will sum to the existing values)
	    gridQuantity, //A real3* with the grid data
	    grid,         //A Grid descriptor corresponding to gridQuantity
	    //qw,  //Optional, a device functor that takes (cell, grid) and returns the quadrature weight of cell. cellVolume() is the default.
	    numberMarkers,
	    cudaStream);


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
namespace uammd{
  template<class Kernel>
  class IBM{
    shared_ptr<Kernel> kernel;
    shared_ptr<System> sys;
  public:

    IBM(shared_ptr<System> sys, shared_ptr<Kernel> kern);

    template<class Grid, class PosIterator,
      class QuantityIterator,
      class GridDataIterator>
    void spread(const PosIterator &pos, const QuantityIterator &v,
		GridDataIterator &gridData,
		Grid grid, int numberParticles, cudaStream_t st = 0);

    template<class Grid,
      class PosIterator, class ResultIterator, class GridQuantityIterator>
    void gather(const PosIterator &pos, const ResultIterator &Jq,
		const GridQuantityIterator &gridData,
		Grid & grid, int numberParticles, cudaStream_t st = 0);

    template<class Grid,
      class PosIterator, class ResultIterator, class GridQuantityIterator,
      class QuadratureWeights>
    void gather(const PosIterator &pos, const ResultIterator &Jq,
		const GridQuantityIterator &gridData,
		Grid & grid, const QuadratureWeights &qw, int numberParticles, cudaStream_t st = 0);

    template<bool is2D, class Grid,
      class PosIterator, class ResultIterator, class GridQuantityIterator,
      class QuadratureWeights>
    void gather(const PosIterator &pos, const ResultIterator &Jq,
		const GridQuantityIterator &gridData,
		Grid & grid, const QuadratureWeights &qw, int numberParticles, cudaStream_t st = 0);

    shared_ptr<Kernel> getKernel(){ return this->kernel;}
    
  };

}

#include"IBM.cu"

#endif
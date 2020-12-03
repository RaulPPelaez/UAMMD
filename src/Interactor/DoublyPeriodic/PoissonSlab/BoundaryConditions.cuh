/*Raul P. Pelaez 2020. Chebyshev Boundary conditions for the  Doubly Periodic Poisson solver. Slab geometry.
In this case they are just trivial and equal to 1 all the time. Maybe this class could be ommited.
*/

#ifndef DOUBLYPERIODIC_POISSONSLAB_BOUNDARYCONDITIONS_CUH
#define DOUBLYPERIODIC_POISSONSLAB_BOUNDARYCONDITIONS_CUH
#include"utils/cufftPrecisionAgnostic.h"
#include "global/defines.h"
#include "utils.cuh"
namespace uammd{
  namespace DPPoissonSlab_ns{
    class TrivialBoundaryConditions{
      int instance;
    public:
      __host__ __device__ TrivialBoundaryConditions(int i):instance(i){
      }

      static real getFirstIntegralFactor(){
	return 1.0;
      }

      static real getSecondIntegralFactor(){
	return 1.0;
      }

      // static __device__ cufftComplex_t<real> getRightHandSide(){
      // 	return cufftComplex_t<real>();
      // }

    };

    template<class BoundaryConditions = TrivialBoundaryConditions>
    class BoundaryConditionsDispatch{
    public:
      __host__ __device__ BoundaryConditions operator()(int i){
	return BoundaryConditions(i);
      }
    };

  }
}
#endif

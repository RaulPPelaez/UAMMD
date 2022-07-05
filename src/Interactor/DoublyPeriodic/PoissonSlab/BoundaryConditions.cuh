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
    class TopBoundaryConditions{
      real k, H;
    public:
      TopBoundaryConditions(real k, real H):k(k),H(H){
      }

      real getFirstIntegralFactor() const{
	return (k!=0)*H;
      }

      real getSecondIntegralFactor() const{
	return k!=0?(H*H*k):(1.0);
      }
    };

    class BottomBoundaryConditions{
      real k, H;
    public:
      BottomBoundaryConditions(real k, real H):k(k),H(H){
      }

      real getFirstIntegralFactor() const{
	return (k!=0)*H;
      }

      real getSecondIntegralFactor() const{
	return k!=0?(-H*H*k):(1.0);
      }
    };

    template<class BoundaryConditions, class Klist>
    class BoundaryConditionsDispatch{
      Klist klist;
      real H;
    public:
      BoundaryConditionsDispatch(Klist klist, real H):klist(klist), H(H){}

      BoundaryConditions operator()(int i) const{
	return BoundaryConditions(klist[i], H);
      }
    };

  }
}
#endif
